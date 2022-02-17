#ifndef SPIRV_CROSS_LLVM_LLVM13_HPP
#define SPIRV_CROSS_LLVM_LLVM13_HPP

#include "spirv_llvm.hpp"
#include "spirv_llvm_expr.hpp"
#include "spirv_llvm_llvm13_KaleidoscopeJIT.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MatrixBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <unordered_map>

using namespace std;

namespace SPIRV_CROSS_NAMESPACE
{

struct uniform_constant_information
{
	uint32_t set;
	uint32_t binding;
	const llvm_expr_global_variable *spir_variable;
};

static llvm::ExitOnError ExitOnErr;

class CompilerLLVM::CompilerLLVM_impl : public llvm_expr_codegenerator
{
	using global_variables_t = std::unordered_map<uint32_t, std::shared_ptr<llvm_expr>>;
	using local_variables_t = std::unordered_map<uint32_t, std::shared_ptr<llvm_expr>>;
	using functions_t = std::unordered_map<string, std::shared_ptr<llvm_expr_function>>;
	using function_prototypes_t = std::unordered_map<string, std::shared_ptr<llvm_expr_function_prototype>>;

public:
	CompilerLLVM_impl() = delete;
	CompilerLLVM_impl(CompilerLLVM &parent)
	    : m_parent(parent)
	{
	}

	virtual ~CompilerLLVM_impl()
	{
		// Delete the anonymous expression module from the JIT.
		ExitOnErr(m_jit_rt->remove());
	}

	void reset();
	string get_llvm_string();
	bool validate_function(functions_t::key_type function_id, std::string &msg);
	void init(bool enable_optimizations, spv::ExecutionModel execution_model);

	llvm_expr *find_variable(uint32_t id);
	std::shared_ptr<llvm_expr> find_variable_and_share(uint32_t id);
	llvm_expr_function *find_function(functions_t::key_type id);
	llvm_expr_function_prototype *find_function_prototype(functions_t::key_type id);

	void codegen_store(const llvm_expr &ptr, const llvm_expr &variable);
	void codegen_load(const llvm_expr &ptr, std::shared_ptr<llvm_expr> lhe);

	void codegen_shader_input_loads(llvm_expr_local_variable *entry_point_input_pointer,
	                                const vector<llvm_expr_global_variable *> &input_variables,
	                                const vector<uint32_t> &locations);
	void codegen_shader_resource_descriptors_loads(const llvm_expr_local_variable *entry_point_descriptors_table,
	                                               const vector<uniform_constant_information> &descriptor_information);
	void codegen_vertex_shader_varyings_store(llvm_expr_local_variable *entry_point_output_argument,
	                                          const vector<llvm_expr_global_variable *> &output_variables,
	                                          const vector<uint32_t> &locations);
	void codegen_fragment_shader_color_attachment_stores(llvm_expr_local_variable *entry_point_output_argument,
	                                                     const vector<llvm_expr_global_variable *> &output_variables,
	                                                     const vector<uint32_t> &locations);

	// SPIRType to llvm::Type vonversions
	llvm::Type *spir_to_llvm_type_basic(SPIRType::BaseType type);
	llvm::Type *spir_to_llvm_type_struct(const SPIRType &spir_type, const std::string &name = string(""));
	llvm::Type *spir_to_llvm_type_vector(const SPIRType &spir_type, const std::string &name = string(""));
	llvm::Type *spir_to_llvm_type_matrix(const SPIRType &spir_type, const std::string &name = string(""));
	llvm::Type *spir_to_llvm_type(const SPIRType &spir_type, const std::string &name = string(""));

	// Implements llvm_expr_codegenerator pure virtual functions
	llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_local_variable> variable) override;
	llvm::GlobalVariable *llvm_expr_codegen(shared_ptr<llvm_expr_global_variable> variable) override;
	llvm::GlobalVariable *llvm_expr_codegen(shared_ptr<llvm_expr_uniform_variables> variable) override;
	llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_constant> constant) override;
	llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_composite> composite) override;
	llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_composite_extract> extract) override;
	llvm::Function *llvm_expr_codegen(shared_ptr<llvm_expr_function_prototype> func_proto) override;
	llvm::Function *llvm_expr_codegen(shared_ptr<llvm_expr_function> func) override;
	llvm::GetElementPtrInst *llvm_expr_codegen(shared_ptr<llvm_expr_access_chain> access_chain) override;
	llvm::CallInst *llvm_expr_codegen(shared_ptr<llvm_expr_function_call> call) override;
	llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_type_cast> cast) override;
	llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_add> add) override;
	llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_sub> sub) override;
	llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_mul> mul) override;
	llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_div> div) override;
	llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_matrix_mult> mul) override;

	void *jit_compile(const string &entry_point_name);

	void emit_return_void()
	{
		m_llvm_builder->CreateRetVoid();
	}

private:
	llvm::Constant *llvm_expr_codegen_num_literal(llvm_expr_constant &constant, uint32_t col, uint32_t row,
	                                              const string &name = string(""));
	llvm::AllocaInst *create_llvm_alloca(const llvm_expr &variable);
	llvm::Constant *construct_int32_immediate(int32_t val);
	llvm::Constant *construct_f32_immediate(float val);
	llvm::GlobalVariable *codegen_global_variable(const SPIRType &spir_type, const string &name);

	llvm::Value *llvm_expr_codegen_const_vector(llvm_expr_constant &constant);
	llvm::Value *llvm_expr_codegen_const_matrix(llvm_expr_constant &constant);
	llvm::Value *llvm_expr_codegen_vector(const llvm_expr_composite &composite);
	llvm::Value *llvm_expr_codegen_matrix(llvm_expr_composite &composite);

	size_t calc_alignment(const SPIRType &type);
	llvm::Value *gep_matrix_column(const llvm_expr &matrix, uint32_t column);
	void codegen_store_matrix(const llvm_expr &ptr, const llvm_expr &variable);
	void codegen_load_matrix(const llvm_expr &ptr, const llvm_expr &variable);

	void add_global_variable(global_variables_t::key_type key, global_variables_t::mapped_type type);
	void add_local_variable(local_variables_t::key_type key, local_variables_t::mapped_type type);
	void add_function(functions_t::key_type id, functions_t::mapped_type function);
	void add_function_prototype(function_prototypes_t::key_type id, function_prototypes_t::mapped_type function);

	// Parent compiler
	CompilerLLVM &m_parent;

	// Globals
	global_variables_t m_global_variables;

	// Local variables
	local_variables_t m_local_variables;

	// All functions
	functions_t m_functions;
	function_prototypes_t m_function_prototypes;

	// LLVM context
	std::unique_ptr<llvm::LLVMContext> m_llvm_context;
	std::unique_ptr<llvm::Module> m_llvm_module;
	std::unique_ptr<llvm::IRBuilder<>> m_llvm_builder;
	std::unique_ptr<llvm::MatrixBuilder<llvm::IRBuilder<>>> m_llvm_matrix_builder;

	// JIT stuff
	std::unique_ptr<llvm::orc::KaleidoscopeJIT> m_jit_compiler;
	llvm::orc::ResourceTrackerSP m_jit_rt;
};

} // namespace SPIRV_CROSS_NAMESPACE

#endif /* SPIRV_CROSS_LLVM_LLVM13_HPP */
