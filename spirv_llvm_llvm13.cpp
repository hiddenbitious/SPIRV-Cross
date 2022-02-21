#include "spirv_llvm_llvm13.hpp"

#include "llvm-c/Core.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"

#include <iostream>

using namespace SPIRV_CROSS_NAMESPACE;

void CompilerLLVM::CompilerLLVM_impl::init(bool enable_optimizations, spv::ExecutionModel execution_model)
{
	if (!::LLVMIsMultithreaded())
		SPIRV_CROSS_THROW("LLVM is not multithreaded");

	string module_name;
	switch (execution_model)
	{
	case spv::ExecutionModelVertex:
		module_name = string("vertex.ll");
		break;

	case spv::ExecutionModelFragment:
		module_name = string("fragment.ll");
		break;
	default:
		assert(0);
	}

	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();
	llvm::InitializeNativeTargetAsmParser();

	m_jit_compiler = ExitOnErr(llvm::orc::KaleidoscopeJIT::Create(enable_optimizations));

	m_llvm_context = unique_ptr<llvm::LLVMContext>(new llvm::LLVMContext());
	m_llvm_module = unique_ptr<llvm::Module>(new llvm::Module(module_name.c_str(), *m_llvm_context));
	m_llvm_module->setDataLayout(m_jit_compiler->getDataLayout());
	m_llvm_builder = unique_ptr<llvm::IRBuilder<>>(new llvm::IRBuilder<>(*m_llvm_context));
	m_llvm_matrix_builder =
	    unique_ptr<llvm::MatrixBuilder<llvm::IRBuilder<>>>(new llvm::MatrixBuilder<llvm::IRBuilder<>>(*m_llvm_builder));
}

void CompilerLLVM::CompilerLLVM_impl::reset()
{
	m_global_variables.clear();
	m_local_variables.clear();
	m_functions.clear();
}

std::string CompilerLLVM::CompilerLLVM_impl::get_llvm_string()
{
	// Print IR to a string
	std::string Str;
	llvm::raw_string_ostream OS(Str);
	OS << *m_llvm_module;
	OS.flush();

	return Str;
}

void *CompilerLLVM::CompilerLLVM_impl::jit_compile(const string &entry_point_name)
{
	const char *libri_glsl_so_path = SPIRV_CROSS_BUILD_DIR "/source/hw/glsl/libri_glsl.so";
	llvm::sys::DynamicLibrary::LoadLibraryPermanently(libri_glsl_so_path);

	// Create a ResourceTracker to track JIT'd memory allocated to our
	// anonymous expression -- that way we can free it after executing.
	m_jit_rt = m_jit_compiler->getMainJITDylib().createResourceTracker();

	auto TSM = llvm::orc::ThreadSafeModule(std::move(m_llvm_module), std::move(m_llvm_context));
	ExitOnErr(m_jit_compiler->addModule(std::move(TSM), m_jit_rt));

	// Get the anonymous expression's JITSymbol.
	auto Sym = ExitOnErr(m_jit_compiler->lookup(entry_point_name));

	// Get the symbol's address and cast it to the right type (takes no
	// arguments, returns a double) so we can call it as a native function.
	void *FP = (void *)(intptr_t)Sym.getAddress();
	//   fprintf(stderr, "Evaluated to %f\n", FP());

	return FP;
}

llvm::Constant *CompilerLLVM::CompilerLLVM_impl::construct_int32_immediate(int32_t val)
{
	return llvm::ConstantInt::get(*m_llvm_context, llvm::APInt(32, val, true));
}

llvm::Constant *CompilerLLVM::CompilerLLVM_impl::construct_f32_immediate(float val)
{
	return llvm::ConstantFP::get(*m_llvm_context, llvm::APFloat(val));
}

bool CompilerLLVM::CompilerLLVM_impl::validate_function(functions_t::key_type function_id, std::string &msg)
{
	// Verify function
	llvm::raw_string_ostream errors_stream(msg);

	llvm_expr_function *func = find_function(function_id);
	if (llvm::verifyFunction(*static_cast<llvm::Function *>(func->get_value()), &errors_stream))
	{
		cout << "Function verification failed." << endl;
		cout << "-------------" << endl;
		cout << errors_stream.str();
		cout << "-------------" << endl;
		return false;
	}

	return true;
}

static bool is_pow2(uint32_t v)
{
	return v && !(v & (v - 1));
}

static uint32_t round_to_next_pow2(uint32_t v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	return v + 1;
}

size_t CompilerLLVM::CompilerLLVM_impl::calc_alignment(const SPIRType &type)
{
	if (type.basetype == SPIRType::BaseType::Struct)
	{
		return 16;
	}
	else
	{
		uint32_t alignment = 4 * type.vecsize * type.columns;
		return is_pow2(alignment) ? alignment : round_to_next_pow2(alignment);
	}
}

llvm::Type *CompilerLLVM::CompilerLLVM_impl::spir_to_llvm_type_basic(SPIRType::BaseType type)
{
	switch (type)
	{
	case SPIRType::Int:
	case SPIRType::UInt:
		return llvm::Type::getInt32Ty(*m_llvm_context);
	case SPIRType::Int64:
	case SPIRType::UInt64:
		return llvm::Type::getInt64Ty(*m_llvm_context);
	case SPIRType::Float:
		return llvm::Type::getFloatTy(*m_llvm_context);
	case SPIRType::Void:
		return llvm::Type::getVoidTy(*m_llvm_context);

	case SPIRType::SampledImage:
		return llvm::Type::getVoidTy(*m_llvm_context);

	case SPIRType::Half:
	case SPIRType::Boolean:
	case SPIRType::Char:
	case SPIRType::AtomicCounter:
	case SPIRType::Double:
	case SPIRType::Struct:
	case SPIRType::Image:
	case SPIRType::Sampler:
		assert(0 && "Not supported");
		return llvm::Type::getVoidTy(*m_llvm_context);

	case SPIRType::Unknown:
	default:
		assert(0 && "Unkown/bad type");
		return llvm::Type::getVoidTy(*m_llvm_context);
	}
}

llvm::Type *CompilerLLVM::CompilerLLVM_impl::spir_to_llvm_type_struct(const SPIRType &spir_type,
                                                                      const std::string &name)
{
	assert(spir_type.basetype == SPIRType::BaseType::Struct);
	assert(spir_type.member_types.size());

	vector<llvm::Type *> element_types;
	for (const uint32_t element : spir_type.member_types)
	{
		const SPIRType spir_type = m_parent.get<SPIRType>(element);
		llvm::Type *element_type = spir_to_llvm_type(spir_type, name);
		assert(element_type);
		element_types.push_back(element_type);
	}

	llvm::Type *struct_type = llvm::StructType::create(*m_llvm_context, element_types, name, false);
	assert(struct_type);

	return struct_type;
}

llvm::Type *CompilerLLVM::CompilerLLVM_impl::spir_to_llvm_type_vector(const SPIRType &spir_type,
                                                                      const std::string &name)
{
	llvm::Type *llvm_type = spir_to_llvm_type_basic(spir_type.basetype);

	llvm::VectorType *llvm_vector = llvm::VectorType::get(llvm_type, static_cast<unsigned>(spir_type.vecsize), false);

	if (spir_type.pointer)
	{
		assert(spir_type.pointer_depth);
		return llvm::PointerType::get(llvm_vector, 0);
	}
	else
	{
		return llvm_vector;
	}
}

llvm::Type *CompilerLLVM::CompilerLLVM_impl::spir_to_llvm_type(const SPIRType &spir_type, const std::string &name)
{
	if (spir_type.basetype == SPIRType::BaseType::Struct)
	{
		return spir_to_llvm_type_struct(spir_type, name);
	}
	else
	{
		llvm::Type *llvm_type = spir_to_llvm_type_basic(spir_type.basetype);

		if (spir_type.vecsize == 1)
		{
			if (spir_type.pointer)
			{
				assert(spir_type.pointer_depth >= 1);
				llvm::Type *ret_type = llvm::PointerType::get(llvm_type, 0);
				for (uint32_t i = 1; i < spir_type.pointer_depth; ++i)
				{
					ret_type = llvm::PointerType::get(ret_type, 0);
				}
				return ret_type;
			}
			else
			{
				return llvm_type;
			}
		}
		else if (spir_type.columns == 1)
		{
			return spir_to_llvm_type_vector(spir_type, name);
		}
		else
		{
			return spir_to_llvm_type_matrix(spir_type, name);
		}
	}
}

void CompilerLLVM::CompilerLLVM_impl::codegen_store(const llvm_expr &ptr, const llvm_expr &variable)
{
	llvm::Value *llvm_ptr = ptr.get_value();
	assert(llvm_ptr);

	llvm::Value *llvm_variable = variable.get_value();
	assert(llvm_variable);

	m_llvm_builder->CreateStore(llvm_variable, llvm_ptr);
}

void CompilerLLVM::CompilerLLVM_impl::codegen_load(const llvm_expr &ptr, std::shared_ptr<llvm_expr> variable)
{
	llvm::Value *llvm_ptr = ptr.get_value();
	assert(llvm_ptr);

	llvm::LoadInst *load_inst = m_llvm_builder->CreateLoad(llvm_ptr, variable->m_name);
	load_inst->setAlignment(llvm::Align(calc_alignment(variable->m_spir_type)));
	variable->set_value(load_inst);

	add_local_variable(variable->m_id, variable);
}

void CompilerLLVM::CompilerLLVM_impl::codegen_shader_input_loads(
    llvm_expr_local_variable *entry_point_input_pointer, const vector<llvm_expr_global_variable *> &input_variables,
    const vector<uint32_t> &locations)
{
	for (uint32_t i = 0; i < input_variables.size(); ++i)
	{
		llvm::Value *gep_indices = construct_int32_immediate(locations[i]);

		llvm::GetElementPtrInst *gep_inst =
		    llvm::GetElementPtrInst::CreateInBounds(entry_point_input_pointer->get_value(), gep_indices, "attribs");

		assert(m_llvm_builder);
		llvm::Value *val = m_llvm_builder->Insert(gep_inst);

		llvm::Value *loaded_attrib = m_llvm_builder->CreateLoad(val);

		m_llvm_builder->CreateStore(loaded_attrib, input_variables[i]->get_value());
	}
}

void CompilerLLVM::CompilerLLVM_impl::codegen_shader_resource_descriptors_loads(
    const llvm_expr_local_variable *entry_point_descriptors_table,
    const vector<uniform_constant_information> &descriptor_information)
{
	// llvm_expr_function_prototype *func = find_function_prototype("descriptor_debugger");
	// assert(func);

	// vector<llvm::Value *> llvm_args(1, nullptr);
	// llvm::Type *star_type = llvm::PointerType::get(llvm::Type::getVoidTy(*m_llvm_context), 0);

	for (const uniform_constant_information &var : descriptor_information)
	{
		// Gep indices
		vector<llvm::Value *> gep_indices(1, nullptr);

		// Offset into the descriptor table to appropriate descriptor set
		gep_indices[0] = construct_int32_immediate(var.set);

		llvm::GetElementPtrInst *gep_into_desc_set_ptr = llvm::GetElementPtrInst::CreateInBounds(
		    entry_point_descriptors_table->get_value(), gep_indices, "__descriptor_table__");
		assert(gep_into_desc_set_ptr);

		llvm::Value *desc_set_ptr_val = m_llvm_builder->Insert(gep_into_desc_set_ptr);

		// Load pointer to descriptor set
		llvm::Value *loaded_desc_set_ptr = m_llvm_builder->CreateLoad(desc_set_ptr_val);

		// {
		// 	llvm_args[0] = m_llvm_builder->CreateBitCast(loaded_desc_set_ptr, star_type);
		// 	llvm::CallInst *llvm_call =
		// 	    m_llvm_builder->CreateCall(static_cast<llvm::Function *>(func->get_value()), llvm_args);
		// }

		gep_indices[0] = construct_int32_immediate(0);
		llvm::GetElementPtrInst *gep_into_set =
		    llvm::GetElementPtrInst::CreateInBounds(loaded_desc_set_ptr, gep_indices, "__descriptor_table__");
		assert(gep_into_set);

		llvm::Value *set_star = m_llvm_builder->Insert(gep_into_set);

		// Load descriptor set
		llvm::Value *loaded_set = m_llvm_builder->CreateLoad(set_star);

		// {
		// 	llvm_args[0] = m_llvm_builder->CreateBitCast(loaded_set, star_type);
		// 	llvm::CallInst *llvm_call =
		// 	    m_llvm_builder->CreateCall(static_cast<llvm::Function *>(func->get_value()), llvm_args);
		// }

		// Now offset inside the set to the appropriate binding
		gep_indices[0] = construct_int32_immediate(var.binding);
		llvm::GetElementPtrInst *gep_into_binding =
		    llvm::GetElementPtrInst::CreateInBounds(loaded_set, gep_indices, "__descriptor_table__");
		assert(gep_into_binding);

		llvm::Value *binding_val = m_llvm_builder->Insert(gep_into_binding);

		// Load the actual descriptor
		llvm::Value *loaded_descriptor = m_llvm_builder->CreateLoad(binding_val);

		// {
		// 	llvm_args[0] = m_llvm_builder->CreateBitCast(loaded_descriptor, star_type);
		// 	llvm::CallInst *llvm_call =
		// 	    m_llvm_builder->CreateCall(static_cast<llvm::Function *>(func->get_value()), llvm_args);
		// }

		// One last load for uniform buffers
		if (var.spir_variable->m_spir_type.basetype == SPIRType::BaseType::Struct)
		{
			gep_indices[0] = construct_int32_immediate(0);
			llvm::GetElementPtrInst *last_gep =
			    llvm::GetElementPtrInst::CreateInBounds(loaded_descriptor, gep_indices, "__descriptor_table__");
			llvm::Value *last_star = m_llvm_builder->Insert(last_gep);

			loaded_descriptor = m_llvm_builder->CreateLoad(last_star);

			// Store it in the shader attribute variable
			llvm::Value *shader_uniform = var.spir_variable->get_value();
			assert(shader_uniform);

			llvm::Type *double_star_type = llvm::PointerType::get(llvm::Type::getVoidTy(*m_llvm_context), 0);
			double_star_type = llvm::PointerType::get(double_star_type, 0);
			llvm::Value *casted_shader_uniform_ptr = m_llvm_builder->CreateBitCast(shader_uniform, double_star_type);
			m_llvm_builder->CreateStore(loaded_descriptor, casted_shader_uniform_ptr);
		}
		else
		{
			// Store it in the shader attribute variable
			llvm::Value *shader_uniform = var.spir_variable->get_value();
			assert(shader_uniform);

			llvm::Type *double_star_type = llvm::PointerType::get(llvm::Type::getVoidTy(*m_llvm_context), 0);
			double_star_type = llvm::PointerType::get(llvm::PointerType::get(double_star_type, 0), 0);
			llvm::Value *casted_shader_uniform_ptr = m_llvm_builder->CreateBitCast(shader_uniform, double_star_type);
			m_llvm_builder->CreateStore(loaded_descriptor, casted_shader_uniform_ptr);
		}

		// {
		// 	llvm_args[0] = m_llvm_builder->CreateBitCast(loaded_descriptor, star_type);
		// 	;
		// 	llvm::CallInst *llvm_call =
		// 	    m_llvm_builder->CreateCall(static_cast<llvm::Function *>(func->get_value()), llvm_args);
		// }
	}
}

void CompilerLLVM::CompilerLLVM_impl::codegen_vertex_shader_varyings_store(
    llvm_expr_local_variable *entry_point_output_argument, const vector<llvm_expr_global_variable *> &output_variables,
    const vector<uint32_t> &locations)
{
	// Copy gl_Position
	{
		// Gep indices
		// gl_Position appears to always the first element of gl_PerVertex
		vector<llvm::Value *> gl_Pervertex_gep_indices(2, construct_int32_immediate(0));
		llvm::Value *position_gep_index = construct_int32_immediate(0);
		llvm::Value *per_vertex = output_variables[0]->get_value();
		assert(per_vertex);

		// Get gl_Position from inside gl_PerVertex. It appears that it is always the first element
		llvm::GetElementPtrInst *gl_position_gep =
		    llvm::GetElementPtrInst::CreateInBounds(per_vertex, gl_Pervertex_gep_indices, "gl_Position");
		assert(gl_position_gep);

		llvm::Value *gl_position_ptr = m_llvm_builder->Insert(gl_position_gep);
		assert(gl_position_ptr);

		// Do a gep instruction to get the pointer to the varying output we want
		llvm::GetElementPtrInst *varying_out_gep = llvm::GetElementPtrInst::CreateInBounds(
		    entry_point_output_argument->get_value(), position_gep_index, "varyings");
		assert(varying_out_gep);

		llvm::Value *varying_out_ptr = m_llvm_builder->Insert(varying_out_gep);
		assert(varying_out_ptr);

		// Load varying value
		llvm::Value *varying_val = m_llvm_builder->CreateLoad(gl_position_ptr);
		assert(varying_val);

		// Store it in the varying pointer
		m_llvm_builder->CreateStore(varying_val, varying_out_ptr);
	}

	// Now copy rest of varyings
	for (uint32_t i = 1; i < output_variables.size(); ++i)
	{
		llvm::Value *varying_gep_index = construct_int32_immediate(locations[i] + 1);

		// Do a gep instruction to get the pointer to the varying output we want
		llvm::GetElementPtrInst *varying_out_gep = llvm::GetElementPtrInst::CreateInBounds(
		    entry_point_output_argument->get_value(), varying_gep_index, "varyings");
		assert(varying_out_gep);

		llvm::Value *varying_out_ptr = m_llvm_builder->Insert(varying_out_gep);
		assert(varying_out_ptr);

		llvm::Value *var = output_variables[i]->get_value();
		assert(var);

		// Load varying value
		llvm::Value *varying_val = m_llvm_builder->CreateLoad(var);
		assert(varying_val);

		// Store it in the varying pointer
		m_llvm_builder->CreateStore(varying_val, varying_out_ptr);
	}
}

void CompilerLLVM::CompilerLLVM_impl::codegen_fragment_shader_color_attachment_stores(
    llvm_expr_local_variable *entry_point_output_argument, const vector<llvm_expr_global_variable *> &output_variables,
    const vector<uint32_t> &locations)
{
	// Gep indices
	vector<llvm::Value *> color_attachment_gep_index(1, construct_int32_immediate(locations[0]));

	for (auto &var : output_variables)
	{
		// GEP in the color attachment entry point argument to find the offset to the appropriate color attachment
		llvm::GetElementPtrInst *color_attachment_gep = llvm::GetElementPtrInst::CreateInBounds(
		    entry_point_output_argument->get_value(), color_attachment_gep_index, "color_attachment");
		assert(color_attachment_gep);

		llvm::Value *varying_out_ptr = m_llvm_builder->Insert(color_attachment_gep);
		assert(varying_out_ptr);

		assert(var->get_value());
		llvm::Value *varying_val = m_llvm_builder->CreateLoad(var->get_value());
		assert(varying_val);

		m_llvm_builder->CreateStore(varying_val, varying_out_ptr);
	}
}

void CompilerLLVM::CompilerLLVM_impl::add_function(functions_t::key_type id, functions_t::mapped_type function)
{
	if (find_function(id) == nullptr)
	{
		m_functions[id] = function;
	}
}

void CompilerLLVM::CompilerLLVM_impl::add_function_prototype(function_prototypes_t::key_type id,
                                                             function_prototypes_t::mapped_type function)
{
	if (find_function_prototype(id) == nullptr)
	{
		m_function_prototypes[id] = function;
	}
}

llvm_expr_function *CompilerLLVM::CompilerLLVM_impl::find_function(functions_t::key_type id)
{
	auto func_iterator = m_functions.find(id);
	return func_iterator != m_functions.end() ? func_iterator->second.get() : nullptr;
}

llvm_expr_function_prototype *CompilerLLVM::CompilerLLVM_impl::find_function_prototype(
    function_prototypes_t::key_type id)
{
	auto func_iterator = m_function_prototypes.find(id);
	return func_iterator != m_function_prototypes.end() ? func_iterator->second.get() : nullptr;
}

llvm_expr *CompilerLLVM::CompilerLLVM_impl::find_variable(uint32_t id)
{
	auto global_variable_iterator = m_global_variables.find(id);
	if (global_variable_iterator != m_global_variables.end())
		return global_variable_iterator->second.get();

	auto local_variable_iterator = m_local_variables.find(id);
	if (local_variable_iterator != m_local_variables.end())
		return local_variable_iterator->second.get();

	return nullptr;
}

std::shared_ptr<llvm_expr> CompilerLLVM::CompilerLLVM_impl::find_variable_and_share(uint32_t id)
{
	auto global_variable_iterator = m_global_variables.find(id);
	if (global_variable_iterator != m_global_variables.end())
		return global_variable_iterator->second;

	auto local_variable_iterator = m_local_variables.find(id);
	if (local_variable_iterator != m_local_variables.end())
		return local_variable_iterator->second;

	return nullptr;
}

void CompilerLLVM::CompilerLLVM_impl::add_global_variable(global_variables_t::key_type key,
                                                          global_variables_t::mapped_type type)
{
	if (m_global_variables.find(key) == m_global_variables.end())
	{
		m_global_variables[key] = std::move(type);
	}
}

void CompilerLLVM::CompilerLLVM_impl::add_local_variable(local_variables_t::key_type key,
                                                         local_variables_t::mapped_type type)
{
	assert(type != nullptr);

	if (m_local_variables.find(key) == m_local_variables.end())
	{
		m_local_variables[key] = std::move(type);
	}
}

llvm::AllocaInst *CompilerLLVM::CompilerLLVM_impl::create_llvm_alloca(const llvm_expr &variable)
{
	assert(variable.m_spir_type.basetype != SPIRType::BaseType::Unknown);

	llvm::AllocaInst *alloca =
	    m_llvm_builder->CreateAlloca(spir_to_llvm_type(variable.m_spir_type), nullptr, variable.m_name);
	alloca->setAlignment(llvm::Align(calc_alignment(variable.m_spir_type)));

	return alloca;
}

llvm::Constant *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen_num_literal(llvm_expr_constant &constant,
                                                                               uint32_t col, uint32_t row,
                                                                               const string &name)
{
	const SPIRConstant &spir_constant = constant.m_spir_constant;
	const SPIRType &spir_type = m_parent.get<SPIRType>(spir_constant.constant_type);

	llvm::Constant *llvm_const = nullptr;

	switch (spir_type.basetype)
	{
	case SPIRType::Int:
		llvm_const = llvm::ConstantInt::get(
		    *m_llvm_context, llvm::APInt(32, static_cast<uint64_t>(spir_constant.scalar_i32(col, row)), true));
		break;
	case SPIRType::UInt:
		llvm_const = llvm::ConstantInt::get(
		    *m_llvm_context, llvm::APInt(32, static_cast<uint64_t>(spir_constant.scalar(col, row)), false));
		break;
	case SPIRType::Int64:
		llvm_const = llvm::ConstantInt::get(
		    *m_llvm_context, llvm::APInt(64, static_cast<uint64_t>(spir_constant.scalar_i64(col, row)), true));
		break;
	case SPIRType::UInt64:
		llvm_const = llvm::ConstantInt::get(
		    *m_llvm_context, llvm::APInt(64, static_cast<uint64_t>(spir_constant.scalar_u64(col, row)), false));
		break;
	case SPIRType::Float:
		llvm_const = llvm::ConstantFP::get(*m_llvm_context, llvm::APFloat(spir_constant.scalar_f32(col, row)));
		break;
	case SPIRType::Double:
		llvm_const = llvm::ConstantFP::get(*m_llvm_context, llvm::APFloat(spir_constant.scalar_f64(col, row)));
		break;
	case SPIRType::Unknown:
	case SPIRType::Void:
	case SPIRType::Boolean:
	case SPIRType::Char:
	case SPIRType::AtomicCounter:
	case SPIRType::Half:
		assert(0);
		return nullptr;
	default:
		assert(0);
		return nullptr;
	}

	constant.set_value(llvm_const);
	return llvm_const;
}

llvm::Value *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen_const_vector(llvm_expr_constant &const_vector)
{
	llvm::Type *llvm_vector_type = spir_to_llvm_type(const_vector.m_spir_type, const_vector.m_name);
	llvm::Value *llvm_vector = llvm::UndefValue::get(llvm_vector_type);

	for (uint32_t col = 0; col < const_vector.m_spir_type.columns; ++col)
	{
		for (uint32_t row = 0; row < const_vector.m_spir_type.vecsize; ++row)
		{
			const uint32_t idx = col * const_vector.m_spir_type.vecsize + row;
			llvm::Constant *llvm_idx = llvm::ConstantInt::get(*m_llvm_context, llvm::APInt(32, idx, true));
			llvm::Value *element = llvm_expr_codegen_num_literal(const_vector, col, row);

			llvm::InsertElementInst *insert_element_instr =
			    llvm::InsertElementInst::Create(llvm_vector, element, llvm_idx, const_vector.m_name);
			m_llvm_builder->Insert(insert_element_instr);

			llvm_vector = insert_element_instr;
		}
	}

	const_vector.set_value(llvm_vector);

	return llvm_vector;
}

llvm::Value *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen_vector(const llvm_expr_composite &composite)
{
	llvm::Type *llvm_vector_type = spir_to_llvm_type_vector(composite.m_spir_type, composite.m_name);
	llvm::Value *llvm_vector = llvm::UndefValue::get(llvm_vector_type);
	llvm::InsertElementInst *insert_element_instr = nullptr;

	uint32_t row = 0;
	for (auto &member : composite.m_members)
	{
		llvm::Constant *llvm_idx = llvm::ConstantInt::get(*m_llvm_context, llvm::APInt(32, row++, true));

		insert_element_instr =
		    llvm::InsertElementInst::Create(llvm_vector, member->get_value(), llvm_idx, composite.m_name);
		m_llvm_builder->Insert(insert_element_instr);

		llvm_vector = insert_element_instr;
	}

	return llvm_vector;
}

llvm::Value *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen(shared_ptr<llvm_expr_local_variable> variable)
{
	llvm_expr_local_variable *existing = static_cast<llvm_expr_local_variable *>(find_variable(variable->m_id));

	if (existing == nullptr)
	{
		variable->set_value(create_llvm_alloca(*variable));
		add_local_variable(variable->m_id, variable);
	}
	else
	{
	}

	return variable->get_value();
}

llvm::GlobalVariable *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen(shared_ptr<llvm_expr_global_variable> variable)
{
	llvm::Type *llvm_type = spir_to_llvm_type(variable->m_spir_type, variable->m_name);
	llvm::Constant *initializer = llvm::ConstantAggregateZero::get(llvm_type);

	variable->m_llvm_global_var = new llvm::GlobalVariable(
	    *m_llvm_module, llvm_type, false, llvm::GlobalValue::CommonLinkage, initializer, variable->m_name);

	variable->m_llvm_global_var->setAlignment(llvm::MaybeAlign(calc_alignment(variable->m_spir_type)));
	add_global_variable(variable->m_id, variable);

	return variable->m_llvm_global_var;
}

llvm::GlobalVariable *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen(
    shared_ptr<llvm_expr_uniform_variables> variable)
{
	llvm::Type *llvm_type = llvm::PointerType::get(spir_to_llvm_type(variable->m_spir_type, variable->m_name), 0);
	llvm::Constant *initializer = llvm::ConstantAggregateZero::get(llvm_type);

	variable->m_llvm_global_var = new llvm::GlobalVariable(
	    *m_llvm_module, llvm_type, false, llvm::GlobalValue::CommonLinkage, initializer, variable->m_name);

	variable->m_llvm_global_var->setAlignment(llvm::MaybeAlign(calc_alignment(variable->m_spir_type)));
	add_global_variable(variable->m_id, variable);

	return variable->m_llvm_global_var;
}

llvm::GetElementPtrInst *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen(
    shared_ptr<llvm_expr_access_chain> access_chain)
{
	vector<llvm::Value *> llvm_indices;

	llvm_indices.push_back(construct_int32_immediate(0));

	const SPIRType *current_type = &access_chain->m_composite.m_spir_type;
	uint32_t member_idx = 0;
	for (uint32_t i = 0; i < access_chain->m_indices.size(); ++i)
	{
		if (current_type->columns == 1)
		{
			llvm_indices.push_back(access_chain->m_indices[i]->codegen(*this));
		}
		else
		{
			assert(access_chain->m_indices.size() - i >= 2);

			const llvm_expr_constant *idx_const_0 = static_cast<llvm_expr_constant *>(access_chain->m_indices[i]);
			const llvm_expr_constant *idx_const_1 = static_cast<llvm_expr_constant *>(access_chain->m_indices[i + 1]);
			++i;

			const uint32_t col = idx_const_0->m_spir_constant.scalar(0, 0);
			const uint32_t row = idx_const_1->m_spir_constant.scalar(0, 0);
			const uint32_t idx = col * current_type->vecsize + row;

			llvm_indices.push_back(construct_int32_immediate(idx));
		}

		if (current_type->member_types.size())
		{
			current_type = &m_parent.get<const SPIRType>(current_type->member_types[member_idx++]);
		}
	}

	llvm::Value *ptr;
	if (access_chain->m_composite.m_spir_type.pointer)
	{
		llvm::LoadInst *load_inst = m_llvm_builder->CreateLoad(access_chain->m_composite.get_value());
		load_inst->setAlignment(llvm::Align(calc_alignment(access_chain->m_spir_type)));
		ptr = load_inst;
	}
	else
	{
		ptr = access_chain->m_composite.get_value();
	}

	llvm::GetElementPtrInst *gep =
	    llvm::GetElementPtrInst::CreateInBounds(ptr, llvm_indices, m_parent.to_name(access_chain->m_composite.m_id));

	m_llvm_builder->Insert(gep, access_chain->m_name);

	access_chain->set_value(gep);
	add_local_variable(access_chain->m_id, access_chain);

	return gep;
}

llvm::Value *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen(shared_ptr<llvm_expr_constant> constant)
{
	if (constant->get_value())
	{
		return constant->get_value();
	}

	llvm::Value *res;
	if (constant->m_spir_type.vecsize == 1)
		res = llvm_expr_codegen_num_literal(*constant, 0, 0, constant->m_name);
	else
		res = llvm_expr_codegen_const_vector(*constant);

	add_local_variable(constant->m_id, constant);

	return res;
}

llvm::Value *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen(shared_ptr<llvm_expr_composite> composite)
{
	if (composite->m_spir_type.vecsize > 1 && composite->m_spir_type.columns == 1)
	{
		composite->set_value(llvm_expr_codegen_vector(*composite));
	}
	else if (composite->m_spir_type.columns > 1)
	{
		composite->set_value(llvm_expr_codegen_matrix(*composite));
	}

	add_local_variable(composite->m_id, composite);

	return composite->get_value();
}

llvm::Function *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen(shared_ptr<llvm_expr_function_prototype> func_proto)
{
	if (func_proto->get_value())
	{
		assert(0);
		return static_cast<llvm::Function *>(func_proto->get_value());
	}

	// Convert SPIR types to llvm types
	vector<llvm::Type *> llvm_arguments;
	for (uint32_t i = 0; i < func_proto->m_arguments.size(); ++i)
	{
		llvm_arguments.push_back(
		    spir_to_llvm_type(func_proto->m_arguments[i]->m_spir_type, func_proto->m_arguments[i]->m_name));
	}

	llvm::Type *ret_type = spir_to_llvm_type(func_proto->m_spir_type);
	llvm::FunctionType *func_type = llvm::FunctionType::get(ret_type, llvm_arguments, false);

	llvm::Function *llvm_func =
	    llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, func_proto->m_name, m_llvm_module.get());

	// Set names to function arguments
	uint32_t i = 0;
	for (llvm::Argument &arg : llvm_func->args())
	{
		func_proto->m_arguments[i]->set_value(&arg);
		arg.setName(func_proto->m_arguments[i++]->m_name);
	}

	func_proto->set_value(llvm_func);
	add_function_prototype(func_proto->m_name, func_proto);

	return llvm_func;
}

llvm::Function *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen(shared_ptr<llvm_expr_function> func)
{
	if (func->get_value())
	{
		assert(0);
		return static_cast<llvm::Function *>(func->get_value());
	}

	assert(func->m_prototype.get_value());
	assert(find_function_prototype(func->m_prototype.m_name));

	llvm::Function *llvm_func = static_cast<llvm::Function *>(func->m_prototype.get_value());

	// Create a new basic block to start insertion into.
	const string enty_name = func->m_name + string("_entry_block");
	func->m_current_llvm_block = llvm::BasicBlock::Create(*m_llvm_context, enty_name, llvm_func);

	func->set_value(func->m_prototype.get_value());
	add_function(func->m_name, func);

	// Add arguments into local variables
	uint32_t i = 0;
	for (auto &arg : func->m_prototype.m_arguments)
	{
		add_local_variable(arg->m_id, arg);
		++i;
	}

	m_llvm_builder->SetInsertPoint(func->m_current_llvm_block);

	return llvm_func;
}

llvm::CallInst *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen(shared_ptr<llvm_expr_function_call> call)
{
	vector<llvm::Value *> llvm_args;
	llvm_args.reserve(call->m_arguments.size());

	for (auto &arg : call->m_arguments)
		llvm_args.push_back(arg->get_value());

	llvm::CallInst *llvm_call =
	    m_llvm_builder->CreateCall(static_cast<llvm::Function *>(call->m_func.get_value()), llvm_args, call->m_name);
	call->set_value(llvm_call);
	add_local_variable(call->m_id, call);

	return llvm_call;
}

llvm::Value *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen(shared_ptr<llvm_expr_type_cast> cast)
{
	llvm::Value *res;
	llvm::Type *llvm_res_type = spir_to_llvm_type(cast->m_spir_type);

	switch (cast->m_spir_type.basetype)
	{
	case SPIRType::BaseType::Float:
	case SPIRType::BaseType::Double:
		switch (cast->m_value.m_spir_type.basetype)
		{
		case SPIRType::BaseType::SByte:
		case SPIRType::BaseType::Short:
		case SPIRType::BaseType::Int:
		case SPIRType::BaseType::Int64:
			res = m_llvm_builder->CreateCast(llvm::Instruction::SIToFP, cast->m_value.get_value(), llvm_res_type);
			break;

		case SPIRType::BaseType::UByte:
		case SPIRType::BaseType::UShort:
		case SPIRType::BaseType::UInt:
		case SPIRType::BaseType::UInt64:
			res = m_llvm_builder->CreateCast(llvm::Instruction::UIToFP, cast->m_value.get_value(), llvm_res_type);
			break;

		default:
			res = nullptr;
			assert(0);
		}
		break;

	default:
		res = nullptr;
		assert(0);
	}

	cast->set_value(res);
	add_local_variable(cast->m_id, cast);

	return res;
}

llvm::Value *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen(shared_ptr<llvm_expr_add> add)
{
	llvm::Value *left = add->m_left.get_value();
	llvm::Value *right = add->m_right.get_value();
	llvm::Value *res;

	switch (add->m_left.m_spir_type.basetype)
	{
	case SPIRType::BaseType::SByte:
	case SPIRType::BaseType::Short:
	case SPIRType::BaseType::Int:
	case SPIRType::BaseType::Int64:
	case SPIRType::BaseType::UByte:
	case SPIRType::BaseType::UShort:
	case SPIRType::BaseType::UInt:
	case SPIRType::BaseType::UInt64:
		res = m_llvm_builder->CreateAdd(left, right, add->m_name);
		break;

	case SPIRType::BaseType::Float:
	case SPIRType::BaseType::Double:
		res = m_llvm_builder->CreateFAdd(left, right, add->m_name);
		break;

	default:
		res = nullptr;
		assert(0);
	}

	add->set_value(res);
	add_local_variable(add->m_id, add);

	return res;
}

llvm::Value *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen(shared_ptr<llvm_expr_sub> sub)
{
	llvm::Value *left = sub->m_left.get_value();
	llvm::Value *right = sub->m_right.get_value();
	llvm::Value *res;

	switch (sub->m_left.m_spir_type.basetype)
	{
	case SPIRType::BaseType::SByte:
	case SPIRType::BaseType::Short:
	case SPIRType::BaseType::Int:
	case SPIRType::BaseType::Int64:
	case SPIRType::BaseType::UByte:
	case SPIRType::BaseType::UShort:
	case SPIRType::BaseType::UInt:
	case SPIRType::BaseType::UInt64:
		res = m_llvm_builder->CreateSub(left, right, sub->m_name);
		break;

	case SPIRType::BaseType::Float:
	case SPIRType::BaseType::Double:
		res = m_llvm_builder->CreateFSub(left, right, sub->m_name);
		break;

	default:
		res = nullptr;
		assert(0);
	}

	sub->set_value(res);
	add_local_variable(sub->m_id, sub);

	return res;
}

llvm::Value *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen(shared_ptr<llvm_expr_mul> mul)
{
	llvm::Value *left = mul->m_left.get_value();
	llvm::Value *right = mul->m_right.get_value();
	llvm::Value *res;

	switch (mul->m_left.m_spir_type.basetype)
	{
	case SPIRType::BaseType::SByte:
	case SPIRType::BaseType::Short:
	case SPIRType::BaseType::Int:
	case SPIRType::BaseType::Int64:
	case SPIRType::BaseType::UByte:
	case SPIRType::BaseType::UShort:
	case SPIRType::BaseType::UInt:
	case SPIRType::BaseType::UInt64:
		res = m_llvm_builder->CreateMul(left, right, mul->m_name);
		break;

	case SPIRType::BaseType::Float:
	case SPIRType::BaseType::Double:
		res = m_llvm_builder->CreateFMul(left, right, mul->m_name);
		break;

	default:
		res = nullptr;
		assert(0);
	}

	mul->set_value(res);
	add_local_variable(mul->m_id, mul);

	return res;
}

llvm::Value *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen(shared_ptr<llvm_expr_div> div)
{
	llvm::Value *left = div->m_left.get_value();
	llvm::Value *right = div->m_right.get_value();
	llvm::Value *res;

	switch (div->m_left.m_spir_type.basetype)
	{
	case SPIRType::BaseType::SByte:
	case SPIRType::BaseType::Short:
	case SPIRType::BaseType::Int:
	case SPIRType::BaseType::Int64:
		res = m_llvm_builder->CreateSDiv(left, right, div->m_name);
		break;

	case SPIRType::BaseType::UByte:
	case SPIRType::BaseType::UShort:
	case SPIRType::BaseType::UInt:
	case SPIRType::BaseType::UInt64:
		res = m_llvm_builder->CreateUDiv(left, right, div->m_name);
		break;

	case SPIRType::BaseType::Float:
	case SPIRType::BaseType::Double:
		res = m_llvm_builder->CreateFDiv(left, right, div->m_name);
		break;

	default:
		res = nullptr;
		assert(0);
	}

	div->set_value(res);
	add_local_variable(div->m_id, div);

	return res;
}

llvm::Type *CompilerLLVM::CompilerLLVM_impl::spir_to_llvm_type_matrix(const SPIRType &spir_type,
                                                                      const std::string &name)
{
	llvm::Type *llvm_type = spir_to_llvm_type_basic(spir_type.basetype);
	llvm::VectorType *llvm_matrix_type = llvm::VectorType::get(llvm_type, spir_type.columns * spir_type.vecsize, false);
	return llvm_matrix_type;
}

llvm::Value *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen_matrix(llvm_expr_composite &composite)
{
	assert(composite.m_spir_type.vecsize == composite.m_members[0]->m_spir_type.vecsize);
	assert(composite.m_spir_type.columns == composite.m_members.size());

	llvm::Type *llvm_matrix_type = spir_to_llvm_type(composite.m_spir_type, composite.m_name);
	llvm::Value *matrix = llvm::UndefValue::get(llvm_matrix_type);

	for (uint32_t col = 0; col < composite.m_spir_type.columns; ++col)
	{
		for (uint32_t row = 0; row < composite.m_members[0]->m_spir_type.vecsize; ++row)
		{
			llvm::Constant *llvm_idx = llvm::ConstantInt::get(*m_llvm_context, llvm::APInt(32, row, true));
			llvm::ExtractElementInst *extract_element =
			    llvm::ExtractElementInst::Create(composite.m_members[col]->get_value(), llvm_idx);
			m_llvm_builder->Insert(extract_element);

			const uint32_t idx = col * composite.m_spir_type.vecsize + row;
			llvm_idx = llvm::ConstantInt::get(*m_llvm_context, llvm::APInt(32, idx, true));
			llvm::InsertElementInst *insert_element_instr =
			    llvm::InsertElementInst::Create(matrix, extract_element, llvm_idx, composite.m_name);
			m_llvm_builder->Insert(insert_element_instr);

			matrix = insert_element_instr;
		}
	}

	composite.set_value(matrix);

	return matrix;
}

llvm::Value *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen(shared_ptr<llvm_expr_composite_extract> extract)
{
	assert(extract->m_src_composite.m_spir_type.columns > 1);
	assert(extract->m_indices.size() == 1);

	const int base_idx = extract->m_indices[0] * extract->m_src_composite.m_spir_type.vecsize;
	vector<int> llvm_mask;
	llvm_mask.reserve(extract->m_src_composite.m_spir_type.vecsize);
	for (uint32_t r = 0; r < extract->m_src_composite.m_spir_type.vecsize; ++r)
		llvm_mask.push_back(base_idx + r);

	llvm::Value *llvm_extract = m_llvm_builder->CreateShuffleVector(extract->m_src_composite.get_value(), llvm_mask);
	extract->set_value(llvm_extract);
	add_local_variable(extract->m_id, extract);

	return llvm_extract;
}

llvm::Value *CompilerLLVM::CompilerLLVM_impl::llvm_expr_codegen(shared_ptr<llvm_expr_matrix_mult> mul)
{
	llvm::Value *left = mul->m_left.get_value();
	assert(left);
	llvm::Value *right = mul->m_right.get_value();
	assert(right);

	const SPIRType &left_spir_type = mul->m_left.m_spir_type;
	const SPIRType &right_spir_type = mul->m_right.m_spir_type;

	llvm::CallInst *mult_inst = m_llvm_matrix_builder->CreateMatrixMultiply(
	    left, right, left_spir_type.vecsize, left_spir_type.columns, right_spir_type.columns, mul->m_name);

	mul->set_value(mult_inst);
	add_local_variable(mul->m_id, mul);

	return mult_inst;
}
