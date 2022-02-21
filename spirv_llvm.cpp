/*
 * Copyright 2015-2018 ARM Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>
#include <iostream>
#include <memory>

#include "spirv_llvm.hpp"
#include "spirv_llvm_llvm13.hpp"

using namespace std;
using namespace spv;
using namespace SPIRV_CROSS_NAMESPACE;

namespace SPIRV_CROSS_NAMESPACE
{

static SPIRType copy_type_and_downgrade_pointer(const SPIRType &type)
{
	SPIRType copy_type = type;

	if (copy_type.pointer_depth)
	{
		assert(copy_type.pointer);

		if (--copy_type.pointer_depth == 0)
			copy_type.pointer = false;
	}

	return copy_type;
}

} // namespace SPIRV_CROSS_NAMESPACE

CompilerLLVM::CompilerLLVM(vector<uint32_t> spirv_)
    : CompilerGLSL(move(spirv_))
    , m_pimpl(new CompilerLLVM_impl(*this))
    , m_n_inputs(0)
    , m_n_outputs(0)
    , m_current_id(0)
    , m_bin_fp(nullptr)
{
}

CompilerLLVM::CompilerLLVM(const uint32_t *ir_, size_t word_count)
    : CompilerGLSL(ir_, word_count)
    , m_pimpl(new CompilerLLVM_impl(*this))
    , m_n_inputs(0)
    , m_n_outputs(0)
    , m_current_id(0)
    , m_bin_fp(nullptr)
{
}

CompilerLLVM::CompilerLLVM(const ParsedIR &ir_)
    : CompilerGLSL(ir_)
    , m_pimpl(new CompilerLLVM_impl(*this))
    , m_n_inputs(0)
    , m_n_outputs(0)
    , m_current_id(0)
    , m_bin_fp(nullptr)
{
}

CompilerLLVM::CompilerLLVM(ParsedIR &&ir_)
    : CompilerGLSL(move(ir_))
    , m_pimpl(new CompilerLLVM_impl(*this))
    , m_n_inputs(0)
    , m_n_outputs(0)
    , m_current_id(0)
    , m_bin_fp(nullptr)
{
}

CompilerLLVM::~CompilerLLVM()
{
	if (m_pimpl)
	{
		delete m_pimpl;
	}
}

void CompilerLLVM::emit_default_vertex_shader_variables()
{
	const array<const string, 2> vertex_shader_default_interface_variable_names = { "gl_VertexID", "gl_InstanceID" };

	for (uint32_t i = 0; i < vertex_shader_default_interface_variable_names.size(); ++i)
	{
		SPIRType spir_type;
		spir_type.basetype = SPIRType::BaseType::Int;
		ID id = get_next_id();

		std::shared_ptr<llvm_expr_global_variable> global = std::make_shared<llvm_expr_global_variable>(
		    vertex_shader_default_interface_variable_names[i], id, &spir_type);
		global->codegen(*m_pimpl);
	}
}

void CompilerLLVM::emit_default_fragment_shader_variables()
{
}

void CompilerLLVM::emit_default_shader_variables()
{
	switch (m_spirv_entry_point.model)
	{
	case ExecutionModelVertex:
		emit_default_vertex_shader_variables();
		break;

	case ExecutionModelFragment:
		emit_default_fragment_shader_variables();
		break;

	case ExecutionModelGeometry:
	case ExecutionModelTessellationControl:
	case ExecutionModelTessellationEvaluation:
	case ExecutionModelGLCompute:
		assert(0 && "Unsupported execution model");
		break;

	default:
		assert(0 && "Unknown execution model");
	}
}

void CompilerLLVM::construct_implicit_vertex_entry_point_arguments(vector<SPIRType> &arg_types,
                                                                   vector<string> &arg_names, vector<uint32_t> &arg_ids)
{
	// Vertex attributes
	// <4 x float>* %attribs
	SPIRType attribs_type = {};
	attribs_type.basetype = SPIRType::BaseType::Float;
	attribs_type.vecsize = 4;
	attribs_type.pointer = true;
	attribs_type.pointer_depth = 1;
	arg_types.push_back(attribs_type);
	arg_names.push_back("_attribs_");
	arg_ids.push_back(get_next_id());

	// Descriptor table
	SPIRType descriptor_table;
	descriptor_table.basetype = SPIRType::BaseType::Void;
	descriptor_table.vecsize = 1;
	descriptor_table.pointer = true;
	descriptor_table.pointer_depth = 5;
	arg_types.push_back(descriptor_table);
	arg_names.push_back("__descriptor_table__");
	arg_ids.push_back(get_next_id());

	// Varyings
	// <4 x float>* %__gl_Position
	arg_types.push_back(attribs_type);
	arg_names.push_back("__gl_Position");
	arg_ids.push_back(get_next_id());
}

void CompilerLLVM::construct_implicit_fragment_entry_point_arguments(vector<SPIRType> &arg_types,
                                                                     vector<string> &arg_names,
                                                                     vector<uint32_t> &arg_ids)
{
	// Inputs
	// <4 x float>* %attribs
	SPIRType attribs_type;
	attribs_type.basetype = SPIRType::BaseType::Float;
	attribs_type.vecsize = 4;
	attribs_type.pointer = true;
	attribs_type.pointer_depth = 1;
	arg_types.push_back(attribs_type);
	arg_names.push_back("__varyings_in");
	arg_ids.push_back(get_next_id());

	// Descriptor table
	SPIRType descriptor_table;
	descriptor_table.basetype = SPIRType::BaseType::Void;
	descriptor_table.vecsize = 1;
	descriptor_table.pointer = true;
	descriptor_table.pointer_depth = 5;
	arg_types.push_back(descriptor_table);
	arg_names.push_back("__descriptor_table__");
	arg_ids.push_back(get_next_id());

	// Varyings
	// <4 x float>* %__color_attachment_0
	arg_types.push_back(attribs_type);
	arg_names.push_back("__color_attachment_0");
	arg_ids.push_back(get_next_id());
}

void CompilerLLVM::construct_implicit_entry_point_arguments(vector<SPIRType> &arg_types, vector<string> &arg_names,
                                                            vector<uint32_t> &arg_ids)
{
	switch (m_spirv_entry_point.model)
	{
	case ExecutionModelVertex:
		construct_implicit_vertex_entry_point_arguments(arg_types, arg_names, arg_ids);
		break;

	case ExecutionModelFragment:
		construct_implicit_fragment_entry_point_arguments(arg_types, arg_names, arg_ids);
		break;

	default:
		assert(0);
		break;
	}
}

void CompilerLLVM::emit_entry_point_input_loads(void)
{
	m_n_inputs = 0;
	vector<uint32_t> locations;
	vector<llvm_expr_global_variable *> input_variables;
	vector<uniform_constant_information> descriptor_variables;

	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		if (var.storage == StorageClassInput)
		{
			input_variables.push_back(static_cast<llvm_expr_global_variable *>(m_pimpl->find_variable(var.self)));
			locations.push_back(ir.meta[var.self].decoration.location);
			++m_n_inputs;
		}
		else if (var.storage == StorageClassUniformConstant || var.storage == StorageClassUniform)
		{
			const llvm_expr_global_variable *uniform =
			    static_cast<llvm_expr_global_variable *>(m_pimpl->find_variable(var.self));
			descriptor_variables.push_back(
			    { ir.meta[var.self].decoration.set, ir.meta[var.self].decoration.binding, uniform });
		}
	});

	if (!m_n_inputs && !descriptor_variables.size())
		return;

	llvm_expr_function *main_func = m_pimpl->find_function("main");
	assert(main_func);

	if (m_n_inputs)
	{
		llvm_expr_local_variable *entry_point_input_pointer = nullptr;

		switch (m_spirv_entry_point.model)
		{
		case ExecutionModelVertex:
			entry_point_input_pointer = main_func->m_prototype.m_arguments[0].get();
			assert(entry_point_input_pointer->m_name == "_attribs_");
			break;

		case ExecutionModelFragment:
			entry_point_input_pointer = main_func->m_prototype.m_arguments[0].get();
			assert(entry_point_input_pointer->m_name == "__varyings_in");
			break;
		default:
			assert(0 && "Not supported");
			break;
		}

		m_pimpl->codegen_shader_input_loads(entry_point_input_pointer, input_variables, locations);
	}

	if (descriptor_variables.size())
	{
		const llvm_expr_local_variable *descriptor_table = main_func->m_prototype.m_arguments[1].get();
		assert(descriptor_table->m_name == "__descriptor_table__");
		m_pimpl->codegen_shader_resource_descriptors_loads(descriptor_table, descriptor_variables);
	}
}

void CompilerLLVM::emit_vertex_shader_varyings_store(void)
{
	vector<llvm_expr_global_variable *> output_vars;
	vector<uint32_t> output_var_location;

	m_n_outputs = 0;

	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		if (var.storage == StorageClassOutput)
		{
			llvm_expr_global_variable *varying =
			    static_cast<llvm_expr_global_variable *>(m_pimpl->find_variable(var.self));
			assert(varying);
			output_vars.push_back(varying);
			output_var_location.push_back(ir.meta[var.self].decoration.location);
			++m_n_outputs;
		}
	});

	assert(m_n_outputs >= 1);

	if (m_n_outputs)
	{
		llvm_expr_function *main_func = m_pimpl->find_function("main");
		assert(main_func);

		llvm_expr_local_variable *entry_point_output_argument = main_func->m_prototype.m_arguments[2].get();
		assert(entry_point_output_argument->m_name == "__gl_Position");

		m_pimpl->codegen_vertex_shader_varyings_store(entry_point_output_argument, output_vars, output_var_location);
	}
}

void CompilerLLVM::emit_fragment_shader_color_attachment_stores(void)
{
	vector<uint32_t> color_attachment_locations;
	vector<llvm_expr_global_variable *> color_attachment_variables;

	m_n_outputs = 0;

	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		// Output variables
		if (var.storage == StorageClassOutput)
		{
			llvm_expr_global_variable *attachment =
			    static_cast<llvm_expr_global_variable *>(m_pimpl->find_variable(var.self));
			assert(attachment);
			color_attachment_variables.push_back(attachment);
			assert(ir.meta[var.self].decoration.location == 0 && "Only one color attachment @ 0 is supported");
			color_attachment_locations.push_back(ir.meta[var.self].decoration.location);

			++m_n_outputs;
		}
	});

	if (m_n_outputs > 0)
	{
		llvm_expr_function *main_func = m_pimpl->find_function("main");
		assert(main_func);

		llvm_expr_local_variable *entry_point_output_argument = main_func->m_prototype.m_arguments[2].get();
		assert(entry_point_output_argument->m_name == "__color_attachment_0");

		m_pimpl->codegen_fragment_shader_color_attachment_stores(
		    entry_point_output_argument, color_attachment_variables, color_attachment_locations);
	}
}

void CompilerLLVM::emit_shader_outputs_stores(void)
{
	switch (m_spirv_entry_point.model)
	{
	case ExecutionModelVertex:
		emit_vertex_shader_varyings_store();
		break;

	case ExecutionModelFragment:
		emit_fragment_shader_color_attachment_stores();
		break;

	default:
		assert(0);
		break;
	}
}

void CompilerLLVM::emit_glsl_function_prototypes()
{
	{
		const ID func_id = get_next_id();
		const string func_name = "ImageSampleImplicitLod";
		SPIRType spir_return_type;
		spir_return_type.basetype = SPIRType::BaseType::Float;
		spir_return_type.vecsize = 4;

		vector<std::shared_ptr<llvm_expr_local_variable>> arguments;

		SPIRType image_sampler_ptr;
		image_sampler_ptr.basetype = SPIRType::BaseType::Void;
		image_sampler_ptr.pointer = true;
		image_sampler_ptr.pointer_depth = 1;
		ID id = get_next_id();
		arguments.emplace_back(std::make_shared<llvm_expr_local_variable>("image_sampler", id, &image_sampler_ptr));

		SPIRType texture_coordinates;
		texture_coordinates.basetype = SPIRType::BaseType::Float;
		texture_coordinates.pointer = false;
		texture_coordinates.pointer_depth = 0;
		texture_coordinates.vecsize = 4;
		texture_coordinates.self = get_next_id();
		id = get_next_id();
		arguments.emplace_back(std::make_shared<llvm_expr_local_variable>("tex_coords", id, &texture_coordinates));

		std::shared_ptr<llvm_expr_function_prototype> func =
		    std::make_shared<llvm_expr_function_prototype>(arguments, func_name, func_id, &spir_return_type);
		func->codegen(*m_pimpl);
	}

	{
		const ID func_id = get_next_id();
		const string func_name = "descriptor_debugger";
		SPIRType spir_return_type;
		spir_return_type.basetype = SPIRType::BaseType::Void;
		spir_return_type.vecsize = 1;

		vector<std::shared_ptr<llvm_expr_local_variable>> arguments;

		SPIRType image_sampler_ptr;
		image_sampler_ptr.basetype = SPIRType::BaseType::Void;
		image_sampler_ptr.pointer = true;
		image_sampler_ptr.pointer_depth = 1;
		ID id = get_next_id();
		arguments.emplace_back(std::make_shared<llvm_expr_local_variable>("descriptor", id, &image_sampler_ptr));

		std::shared_ptr<llvm_expr_function_prototype> func =
		    std::make_shared<llvm_expr_function_prototype>(arguments, func_name, func_id, &spir_return_type);
		func->codegen(*m_pimpl);
	}
}

void CompilerLLVM::emit_function_prototype(SPIRFunction &spir_func, const Bitset &)
{
	if (spir_func.self != ir.default_entry_point)
	{
		assert(0 && "When does this happen?");
		add_function_overload(spir_func);
	}

	// What is this for?
	local_variable_names = resource_names;

	// Create llvm types for all arguments
	vector<SPIRType> arg_types;
	vector<string> arg_names;
	vector<uint32_t> arg_ids;

	// If this function is the shader's entry point then assign the default arguments we want
	const bool is_entry_point = spir_func.self == ir.default_entry_point;
	if (is_entry_point)
	{
		// It should be empty
		assert(spir_func.arguments.size() == 0);

		construct_implicit_entry_point_arguments(arg_types, arg_names, arg_ids);
	}
	else
	{
		for (auto &arg : spir_func.arguments)
		{
			add_local_variable_name(arg.id);

			arg_types.push_back(get<SPIRType>(arg.type));
			auto *var = maybe_get<SPIRVariable>(arg.id);
			arg_ids.push_back(arg.id);

			if (var)
			{
				arg_names.push_back(to_name(var->self));
				// Hold a pointer to the parameter so we can invalidate the readonly field if needed.
				var->parameter = &arg;
			}
			else
			{
				arg_names.push_back("unnamed_argument");
			}
		}
	}

	vector<std::shared_ptr<llvm_expr_local_variable>> arguments;
	for (uint32_t i = 0; i < arg_types.size(); ++i)
	{
		arguments.emplace_back(std::make_shared<llvm_expr_local_variable>(arg_names[i], arg_ids[i], &arg_types[i]));
	}

	std::shared_ptr<llvm_expr_function_prototype> func_proto = std::make_shared<llvm_expr_function_prototype>(
	    arguments, to_name(spir_func.self), spir_func.self, &get<SPIRType>(spir_func.return_type));
	func_proto->codegen(*m_pimpl);

	std::shared_ptr<llvm_expr_function> func = std::make_shared<llvm_expr_function>(*func_proto, spir_func.self);
	func->codegen(*m_pimpl);
}

void CompilerLLVM::emit_instruction(const Instruction &instruction)
{
	const uint32_t *ops = stream(instruction);
	const Op opcode = static_cast<Op>(instruction.op);
	uint32_t length = instruction.length;

	switch (opcode)
	{
	case OpAccessChain:
	{
		const uint32_t result_type{ ops[0] };
		const uint32_t result_var{ ops[1] };
		const uint32_t composite{ ops[2] };
		const uint32_t *indices{ &ops[3] };

		vector<llvm_expr *> llvm_indices;
		llvm_indices.reserve(length - 3);

		for (uint32_t i = 0; i < length - 3; ++i)
		{
			llvm_indices.push_back(m_pimpl->find_variable(indices[i]));
		}

		llvm_expr *llvm_composite = m_pimpl->find_variable(composite);
		assert(llvm_composite);

		const SPIRType &spir_type = get<SPIRType>(result_type);
		std::shared_ptr<llvm_expr_access_chain> access_chain = std::make_shared<llvm_expr_access_chain>(
		    *llvm_composite, std::move(llvm_indices), to_name(result_var), result_var, &spir_type);
		access_chain->codegen(*m_pimpl);
	}
	break;

	case OpLoad:
	{
		const uint32_t result_type{ ops[0] };
		const uint32_t result{ ops[1] };
		const uint32_t ptr{ ops[2] };

		std::shared_ptr<llvm_expr_local_variable> lhe =
		    std::make_shared<llvm_expr_local_variable>(to_name(result), result, &get<SPIRType>(result_type));

		llvm_expr *llvm_ptr = m_pimpl->find_variable(ptr);
		assert(llvm_ptr);

		m_pimpl->codegen_load(*llvm_ptr, lhe);
	}
	break;

	case OpStore:
	{
		const uint32_t ptr{ ops[0] };
		const uint32_t object{ ops[1] };

		const llvm_expr *lhe_ptr = m_pimpl->find_variable(ptr);
		assert(lhe_ptr);

		const llvm_expr_local_variable *rhe = static_cast<llvm_expr_local_variable *>(m_pimpl->find_variable(object));
		assert(rhe);

		m_pimpl->codegen_store(*lhe_ptr, *rhe);
	}
	break;

	case OpCompositeConstruct:
	{
		const uint32_t result_type{ ops[0] };
		const uint32_t result{ ops[1] };

		vector<llvm_expr *> members;
		members.reserve(length - 2);

		for (uint32_t i = 2; i < length; ++i)
		{
			members.emplace_back(m_pimpl->find_variable(ops[i]));
		}

		const SPIRType &composite_type = get<SPIRType>(result_type);
		std::shared_ptr<llvm_expr_composite> composite =
		    std::make_shared<llvm_expr_composite>(members, to_name(result), result, &composite_type);
		composite->codegen(*m_pimpl);
	}
	break;

	case OpCompositeExtract:
	{
		const uint32_t extract_type{ ops[0] };
		const uint32_t result{ ops[1] };
		const uint32_t composite{ ops[2] };

		vector<uint32_t> indices;
		indices.reserve(length - 3);

		for (uint32_t i = 3; i < length; ++i)
		{
			indices.push_back(ops[i]);
		}

		llvm_expr *llvm_composite = m_pimpl->find_variable(composite);
		assert(llvm_composite);

		const SPIRType &extract_spir_type = get<SPIRType>(extract_type);
		std::shared_ptr<llvm_expr_composite_extract> extract = std::make_shared<llvm_expr_composite_extract>(
		    *llvm_composite, indices, to_name(result), result, &extract_spir_type);
		extract->codegen(*m_pimpl);
	}
	break;

	case OpImageSampleImplicitLod:
	{
		const uint32_t res_id{ ops[1] };
		const uint32_t cis_id{ ops[2] };
		const uint32_t coords_id{ ops[3] };

		llvm_expr_function_prototype *func = m_pimpl->find_function_prototype("ImageSampleImplicitLod");
		assert(func);

		vector<llvm_expr *> arguments;
		arguments.push_back(m_pimpl->find_variable(cis_id));
		arguments.push_back(m_pimpl->find_variable(coords_id));

		std::shared_ptr<llvm_expr_function_call> func_cal =
		    std::make_shared<llvm_expr_function_call>(*func, arguments, to_name(res_id), res_id);
		func_cal->codegen(*m_pimpl);
	}
	break;

	case OpFAdd:
	case OpFSub:
	case OpFMul:
	case OpFDiv:
	case OpIAdd:
	case OpISub:
	case OpIMul:
	case OpSDiv:
	case OpUDiv:
	{
		const uint32_t res_type{ ops[0] };
		const uint32_t res_id{ ops[1] };
		const uint32_t left{ ops[2] };
		const uint32_t right{ ops[3] };

		llvm_expr *llvm_left = m_pimpl->find_variable(left);
		assert(llvm_left);
		llvm_expr *llvm_right = m_pimpl->find_variable(right);
		assert(llvm_right);

		const SPIRType &spir_type = get<SPIRType>(res_type);

		if (opcode == OpFAdd || opcode == OpIAdd)
		{
			std::shared_ptr<llvm_expr_add> add =
			    std::make_shared<llvm_expr_add>(llvm_left, llvm_right, to_name(res_id), res_id, &spir_type);
			add->codegen(*m_pimpl);
		}
		else if (opcode == OpFSub || opcode == OpISub)
		{
			std::shared_ptr<llvm_expr_sub> sub =
			    std::make_shared<llvm_expr_sub>(llvm_left, llvm_right, to_name(res_id), res_id, &spir_type);
			sub->codegen(*m_pimpl);
		}
		else if (opcode == OpFMul || opcode == OpIMul)
		{
			std::shared_ptr<llvm_expr_mul> mul =
			    std::make_shared<llvm_expr_mul>(llvm_left, llvm_right, to_name(res_id), res_id, &spir_type);
			mul->codegen(*m_pimpl);
		}
		else if (opcode == OpFDiv || opcode == OpSDiv || opcode == OpUDiv)
		{
			std::shared_ptr<llvm_expr_div> div =
			    std::make_shared<llvm_expr_div>(llvm_left, llvm_right, to_name(res_id), res_id, &spir_type);
			div->codegen(*m_pimpl);
		}
	}
	break;

	case OpMatrixTimesMatrix:
	case OpMatrixTimesVector:
	{
		const uint32_t res_type{ ops[0] };
		const uint32_t res_id{ ops[1] };
		const uint32_t left{ ops[2] };
		const uint32_t right{ ops[3] };

		llvm_expr *llvm_left = m_pimpl->find_variable(left);
		assert(llvm_left);
		llvm_expr *llvm_right = m_pimpl->find_variable(right);
		assert(llvm_right);

		const SPIRType &spir_type = get<SPIRType>(res_type);

		std::shared_ptr<llvm_expr_matrix_mult> mul =
		    std::make_shared<llvm_expr_matrix_mult>(llvm_left, llvm_right, to_name(res_id), res_id, &spir_type);
		mul->codegen(*m_pimpl);
	}
	break;

	case OpConvertFToU:
	case OpConvertFToS:
	case OpConvertSToF:
	case OpConvertUToF:
	{
		const uint32_t res_type{ ops[0] };
		const uint32_t res_id{ ops[1] };
		const uint32_t value{ ops[2] };

		const SPIRType &spir_type = get<SPIRType>(res_type);
		llvm_expr *llvm_value = m_pimpl->find_variable(value);
		assert(llvm_value);

		std::shared_ptr<llvm_expr_type_cast> cast =
		    std::make_shared<llvm_expr_type_cast>(llvm_value, to_name(res_id), res_id, &spir_type);
		cast->codegen(*m_pimpl);
	}
	break;

	// These opcode are expected to be handled internally by the parser and shouldn't appear here
	case OpReturn:
	case OpConstantComposite:
		assert(0);
		break;

	default:
		assert(0 && "OpCode not handled");
		break;
	}
}

void CompilerLLVM::emit_block_instructions(SPIRBlock &block)
{
	current_emitting_block = &block;
	for (Instruction &op : block.ops)
		emit_instruction(op);
	current_emitting_block = nullptr;
}

void CompilerLLVM::emit_block_chain(SPIRBlock &block)
{
	emit_hoisted_temporaries(block.declare_temporary);

	// SPIRBlock::ContinueBlockType continue_type = SPIRBlock::ContinueNone;
	// if (block.continue_block)
	// 	continue_type = continue_block_type(get<SPIRBlock>(block.continue_block));

	// // If we have loop variables, stop masking out access to the variable now.
	// for (auto var : block.loop_variables)
	// 	get<SPIRVariable>(var).loop_variable_enable = true;

	emit_block_instructions(block);
}

void CompilerLLVM::emit_function(SPIRFunction &func, const Bitset &return_flags)
{
	// Avoid potential cycles.
	if (func.active)
		return;

	func.active = true;

	// If we depend on a function, emit that function before we emit our own function.
	ir.for_each_typed_id<SPIRBlock>([&](uint32_t, SPIRBlock &block) {
		for (Instruction &i : block.ops)
		{
			const uint32_t *ops = stream(i);
			Op op = static_cast<Op>(i.op);

			if (op == OpFunctionCall)
			{
				// Recursively emit functions which are called.
				uint32_t id = ops[2];
				emit_function(get<SPIRFunction>(id), ir.meta[ops[1]].decoration.decoration_flags);
			}
		}
	});

	emit_function_prototype(func, return_flags);

	// Emit constants for future reference
	ir.for_each_typed_id<SPIRConstant>([&](uint32_t, const SPIRConstant &constant) {
		SPIRType &spir_type = get<SPIRType>(constant.constant_type);
		std::shared_ptr<llvm_expr_constant> expr_constant =
		    std::make_shared<llvm_expr_constant>(constant, to_name(constant.self), constant.self, &spir_type);
		expr_constant->codegen(*m_pimpl);
	});

	const bool is_entry_point = func.self == m_spirv_entry_point.self;
	if (is_entry_point)
	{
		emit_entry_point_input_loads();
	}

	// Emit local variables
	for (const auto &var_id : func.local_variables)
	{
		SPIRVariable &local_var = get<SPIRVariable>(var_id);
		if (local_var.storage == StorageClassWorkgroup)
		{
			// log_stream << "Local variable " << __LINE__ << endl;
			assert(0);
		}
		else if (local_var.storage == StorageClassPrivate)
		{
			// log_stream << "Local variable " << __LINE__ << endl;
			assert(0);
		}
		else if (local_var.storage == StorageClassFunction && local_var.remapped_variable &&
		         local_var.static_expression)
		{
			// log_stream << "Local variable " << __LINE__ << endl;
			assert(0);
			// No need to declare this variable, it has a static expression.
			local_var.deferred_declaration = false;
		}
		else if (expression_is_lvalue(var_id))
		{
			// std::cout << "emiting: " << local_var.self << std::endl;
			// add it where?
			add_local_variable_name(local_var.self);

			const string var_name = to_name(local_var.self);

			// Downgrade pointer
			SPIRType copy_type = copy_type_and_downgrade_pointer(get<SPIRType>(local_var.basetype));

			// Create alloca for local variable
			shared_ptr<llvm_expr_local_variable> llvm_local_var =
			    std::make_shared<llvm_expr_local_variable>(to_name(var_id), var_id, &copy_type);
			llvm_local_var->codegen(*m_pimpl);
		}
	}

	SPIRBlock &entry_block = get<SPIRBlock>(func.entry_block);
	emit_block_chain(entry_block);

	// Write out shader outputs
	if (is_entry_point)
	{
		emit_shader_outputs_stores();
	}

	m_pimpl->emit_return_void();

	// string unoptimized_ir = m_pimpl->get_llvm_string();
	// std::cout << "-----------------------------------" << std::endl;
	// std::cout << unoptimized_ir << std::endl;
	// std::cout << "-----------------------------------" << std::endl;

	// Verify function
	string msg;
	if (m_pimpl->validate_function(to_name(m_spirv_entry_point.self), msg) == false)
	{
		cout << msg << endl;
		assert(0);
	}
}

void CompilerLLVM::emit_buffer_block_native(const SPIRVariable &var)
{
	auto &type = get<SPIRType>(var.basetype);

	// Block names should never alias, but from HLSL input they kind of can because block types are reused for UAVs ...
	auto buffer_name = to_name(type.self, false);

	std::shared_ptr<llvm_expr_uniform_variables> uniform =
	    std::make_shared<llvm_expr_uniform_variables>(buffer_name, var.self, &type);
	uniform->codegen(*m_pimpl);
}

void CompilerLLVM::emit_buffer_block(const SPIRVariable &var)
{
	auto &type = get<SPIRType>(var.basetype);
	bool ubo_block = var.storage == StorageClassUniform && has_decoration(type.self, DecorationBlock);

	if (flattened_buffer_blocks.count(var.self))
	{
		assert(0);
		// emit_buffer_block_flattened(var);
	}
	else if (is_legacy() || (!options.es && options.version == 130) ||
	         (ubo_block && options.emit_uniform_buffer_as_plain_uniforms))
	{
		assert(0);
		// emit_buffer_block_legacy(var);
	}
	else
		emit_buffer_block_native(var);
}

void CompilerLLVM::emit_interface_block(const SPIRVariable &var)
{
	const SPIRType &spir_type = get<SPIRType>(var.basetype);

	// Downgrade pointer
	SPIRType copy_type = copy_type_and_downgrade_pointer(spir_type);

	// Promote all inputs vectors to vec4
	if (copy_type.vecsize > 1)
	{
		copy_type.vecsize = 4;
	}

	std::shared_ptr<llvm_expr_global_variable> global =
	    std::make_shared<llvm_expr_global_variable>(to_name(var.self), var.self, &copy_type);
	global->codegen(*m_pimpl);
}

void CompilerLLVM::emit_uniform_constants(const SPIRVariable &var)
{
	const SPIRType &spir_type = get<SPIRType>(var.basetype);

	// Downgrade pointer
	SPIRType copy_type = copy_type_and_downgrade_pointer(spir_type);

	std::shared_ptr<llvm_expr_uniform_variables> uniform =
	    std::make_shared<llvm_expr_uniform_variables>(to_name(var.self), var.self, &copy_type);
	uniform->codegen(*m_pimpl);
}

void CompilerLLVM::emit_resources()
{
	// Emit in/out interfaces.
	ir.for_each_typed_id<SPIRVariable>([&](uint32_t, SPIRVariable &var) {
		auto &type = this->get<SPIRType>(var.basetype);

		bool is_hidden = is_hidden_variable(var);

		// Unused output I/O variables might still be required to implement framebuffer fetch.
		if (var.storage == StorageClassOutput && !is_legacy() &&
		    location_is_framebuffer_fetch(get_decoration(var.self, DecorationLocation)) != 0)
		{
			is_hidden = false;
		}

		if (var.storage != StorageClassFunction && type.pointer &&
		    (var.storage == StorageClassInput || var.storage == StorageClassOutput) &&
		    interface_variable_exists_in_entry_point(var.self) && !is_hidden)
		{
			emit_interface_block(var);
		}
		else if (var.storage == StorageClassInput)
		{
			emit_interface_block(var);
		}
		else if (var.storage == StorageClassOutput)
		{
			emit_interface_block(var);
		}
		else if (var.storage == StorageClassUniformConstant)
		{
			emit_uniform_constants(var);
		}

		// Output UBOs and SSBOs
		bool is_block_storage = type.storage == StorageClassStorageBuffer || type.storage == StorageClassUniform ||
		                        type.storage == StorageClassShaderRecordBufferKHR;
		bool has_block_flags = ir.meta[type.self].decoration.decoration_flags.get(DecorationBlock) ||
		                       ir.meta[type.self].decoration.decoration_flags.get(DecorationBufferBlock);

		if (var.storage != StorageClassFunction && type.pointer && is_block_storage && !is_hidden_variable(var) &&
		    has_block_flags)
		{
			emit_buffer_block(var);
		}
	});
}

void CompilerLLVM::emit_header()
{
	emit_default_shader_variables();
	emit_glsl_function_prototypes();
}

string CompilerLLVM::compile()
{
	m_spirv_entry_point = get_entry_point();

	m_pimpl->init(m_llvm_options.optimize_llvm, m_spirv_entry_point.model);

	fixup_type_alias();
	reorder_type_alias();
	build_function_control_flow_graphs_and_analyze();
	update_active_builtins();

	// The third word is the ID bound, start using from this and upwards
	m_current_id = ir.spirv[3];

	uint32_t pass_count = 0;
	do
	{
		if (pass_count >= 3)
			assert(0 && "Over 3 compilation loops detected. Must be a bug!");

		reset();
		m_pimpl->reset();

		emit_header();
		emit_resources();
		emit_function(get<SPIRFunction>(ir.default_entry_point), Bitset());

		pass_count++;
	} while (is_forcing_recompilation());

	string unoptimized_ir = m_pimpl->get_llvm_string();
	m_bin_fp = m_pimpl->jit_compile(to_name(m_spirv_entry_point.self));

	return unoptimized_ir;
}
