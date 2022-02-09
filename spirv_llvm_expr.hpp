#ifndef SPIRV_CROSS_LLVM_EXPR_HPP
#define SPIRV_CROSS_LLVM_EXPR_HPP

#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"

#include <memory>
#include <string>
#include <vector>

using namespace std;

namespace SPIRV_CROSS_NAMESPACE
{

class llvm_expr_codegenerator;

class llvm_expr
{
public:
	llvm_expr(const string &name, uint32_t id, const SPIRType *spir_type = nullptr)
	    : m_name(name)
	    , m_id(id)
	    , m_spir_type(*spir_type)
	    , m_llvm_value(nullptr)
	{
	}

	~llvm_expr()
	{
	}

	llvm_expr &operator=(const llvm_expr &other)
	{
		m_name = other.m_name;
		m_id = other.m_id;
		m_llvm_value = other.m_llvm_value;

		return *this;
	}

	virtual llvm::Value *codegen(llvm_expr_codegenerator &generator) = 0;

	virtual llvm::Value *get_value()
	{
		return m_llvm_value;
	}

	virtual llvm::Value *get_value() const
	{
		return m_llvm_value;
	}

	virtual void set_value(llvm::Value *value)
	{
		m_llvm_value = value;
	}

	string m_name;
	uint32_t m_id;
	const SPIRType m_spir_type;

protected:
	llvm::Value *m_llvm_value;
};

class llvm_expr_local_variable;
class llvm_expr_global_variable;
class llvm_expr_constant;
class llvm_expr_composite;
class llvm_expr_function_prototype;
class llvm_expr_function;
class llvm_expr_access_chain;
class llvm_expr_function_call;
class llvm_expr_type_cast;
class llvm_expr_add;
class llvm_expr_sub;
class llvm_expr_mul;
class llvm_expr_div;

class llvm_expr_codegenerator
{
public:
	virtual llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_local_variable> variable) = 0;
	virtual llvm::GlobalVariable *llvm_expr_codegen(shared_ptr<llvm_expr_global_variable> variable) = 0;
	virtual llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_constant> constant) = 0;
	virtual llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_composite> composite) = 0;
	virtual llvm::Function *llvm_expr_codegen(shared_ptr<llvm_expr_function_prototype> func_proto) = 0;
	virtual llvm::Function *llvm_expr_codegen(shared_ptr<llvm_expr_function> func) = 0;
	virtual llvm::GetElementPtrInst *llvm_expr_codegen(shared_ptr<llvm_expr_access_chain> access_chain) = 0;
	virtual llvm::CallInst *llvm_expr_codegen(shared_ptr<llvm_expr_function_call> call) = 0;
	virtual llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_type_cast> cast) = 0;
	virtual llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_add> add) = 0;
	virtual llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_sub> sub) = 0;
	virtual llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_mul> mul) = 0;
	virtual llvm::Value *llvm_expr_codegen(shared_ptr<llvm_expr_div> div) = 0;
};

class llvm_expr_local_variable : public llvm_expr, public std::enable_shared_from_this<llvm_expr_local_variable>
{
public:
	llvm_expr_local_variable(const string &name, uint32_t id, const SPIRType *spir_type)
	    : llvm_expr(name, id, spir_type)
	{
	}

	llvm::Value *codegen(llvm_expr_codegenerator &generator) override
	{
		return generator.llvm_expr_codegen(shared_from_this());
	}
};

class llvm_expr_global_variable : public llvm_expr, public std::enable_shared_from_this<llvm_expr_global_variable>
{
public:
	llvm_expr_global_variable(const string &name, uint32_t id, const SPIRType *spir_type)
	    : llvm_expr(name, id, spir_type)
	{
	}

	llvm::GlobalVariable *codegen(llvm_expr_codegenerator &generator) override
	{
		return generator.llvm_expr_codegen(shared_from_this());
	}

	llvm::Value *get_value() override
	{
		return static_cast<llvm::Value *>(m_llvm_global_var);
	}

	llvm::Value *get_value() const override
	{
		return static_cast<llvm::Value *>(m_llvm_global_var);
	}

	llvm::GlobalVariable *m_llvm_global_var;
};

class llvm_expr_constant : public llvm_expr, public std::enable_shared_from_this<llvm_expr_constant>
{
public:
	llvm_expr_constant(const SPIRConstant &spir_constant, const string &name, uint32_t id, const SPIRType *spir_type)
	    : llvm_expr(name, id, spir_type)
	    , m_spir_constant(spir_constant)
	{
	}

	llvm::Value *codegen(llvm_expr_codegenerator &generator) override
	{
		return generator.llvm_expr_codegen(shared_from_this());
	}

	const SPIRConstant &m_spir_constant;
};

class llvm_expr_composite : public llvm_expr, public std::enable_shared_from_this<llvm_expr_composite>
{
public:
	llvm_expr_composite(vector<std::shared_ptr<llvm_expr>> &members, const string &name, uint32_t id,
	                    const SPIRType *spir_type)
	    : llvm_expr(name, id, spir_type)
	    , m_members(std::move(members))
	{
	}

	llvm::Value *codegen(llvm_expr_codegenerator &generator) override
	{
		return generator.llvm_expr_codegen(shared_from_this());
	}

	vector<std::shared_ptr<llvm_expr>> m_members;
};

class llvm_expr_function_prototype : public llvm_expr, public std::enable_shared_from_this<llvm_expr_function_prototype>
{
public:
	llvm_expr_function_prototype(vector<std::shared_ptr<llvm_expr_local_variable>> &arguments, const string &name,
	                             uint32_t id, const SPIRType *spir_return_type)
	    : llvm_expr(name, id, spir_return_type)
	    , m_arguments(std::move(arguments))
	{
	}

	llvm::Function *codegen(llvm_expr_codegenerator &generator) override
	{
		return generator.llvm_expr_codegen(shared_from_this());
	}

	vector<std::shared_ptr<llvm_expr_local_variable>> m_arguments;
};

class llvm_expr_function : public llvm_expr, public std::enable_shared_from_this<llvm_expr_function>
{
public:
	llvm_expr_function(llvm_expr_function_prototype &prototype, uint32_t id)
	    : llvm_expr(prototype.m_name, id, &prototype.m_spir_type)
	    , m_prototype(prototype)
	    , m_current_llvm_block(nullptr)
	{
	}

	llvm::Function *codegen(llvm_expr_codegenerator &generator) override
	{
		return generator.llvm_expr_codegen(shared_from_this());
	}

	llvm_expr_function_prototype &m_prototype;
	llvm::BasicBlock *m_current_llvm_block;
};

class llvm_expr_access_chain : public llvm_expr, public std::enable_shared_from_this<llvm_expr_access_chain>
{
public:
	llvm_expr_access_chain(uint32_t base_ptr_id, vector<llvm_expr *> indices, const string &name, uint32_t id,
	                       const SPIRType *spir_type)
	    : llvm_expr(name, id, spir_type)
	    , m_base_ptr_id(base_ptr_id)
	    , m_indices(std::move(indices))
	{
	}

	llvm::GetElementPtrInst *codegen(llvm_expr_codegenerator &generator) override
	{
		return generator.llvm_expr_codegen(shared_from_this());
	}

	uint32_t m_base_ptr_id;
	vector<llvm_expr *> m_indices;
};

class llvm_expr_function_call : public llvm_expr, public std::enable_shared_from_this<llvm_expr_function_call>
{
public:
	llvm_expr_function_call(const llvm_expr_function_prototype &func, vector<llvm_expr *> arguments, const string &name,
	                        uint32_t id)
	    : llvm_expr(name, id, &func.m_spir_type)
	    , m_func(func)
	    , m_arguments(std::move(arguments))
	{
	}

	llvm::CallInst *codegen(llvm_expr_codegenerator &generator) override
	{
		return generator.llvm_expr_codegen(shared_from_this());
	}

	const llvm_expr_function_prototype &m_func;
	vector<llvm_expr *> m_arguments;
};

class llvm_expr_type_cast : public llvm_expr, public std::enable_shared_from_this<llvm_expr_type_cast>
{
public:
	llvm_expr_type_cast(llvm_expr *value, const string &name, uint32_t id, const SPIRType *spir_type)
	    : llvm_expr(name, id, spir_type)
	    , m_value(*value)
	{
	}

	llvm::Value *codegen(llvm_expr_codegenerator &generator) override
	{
		return generator.llvm_expr_codegen(shared_from_this());
	}

	const llvm_expr &m_value;
};

class llvm_expr_add : public llvm_expr, public std::enable_shared_from_this<llvm_expr_add>
{
public:
	llvm_expr_add(llvm_expr *left, llvm_expr *right, const string &name, uint32_t id, const SPIRType *spir_type)
	    : llvm_expr(name, id, spir_type)
	    , m_left(*left)
	    , m_right(*right)
	{
		assert(m_left.m_spir_type.basetype == m_right.m_spir_type.basetype);
	}

	llvm::Value *codegen(llvm_expr_codegenerator &generator) override
	{
		return generator.llvm_expr_codegen(shared_from_this());
	}

	const llvm_expr &m_left, &m_right;
};

class llvm_expr_sub : public llvm_expr, public std::enable_shared_from_this<llvm_expr_sub>
{
public:
	llvm_expr_sub(llvm_expr *left, llvm_expr *right, const string &name, uint32_t id, const SPIRType *spir_type)
	    : llvm_expr(name, id, spir_type)
	    , m_left(*left)
	    , m_right(*right)
	{
		assert(m_left.m_spir_type.basetype == m_right.m_spir_type.basetype);
	}

	llvm::Value *codegen(llvm_expr_codegenerator &generator) override
	{
		return generator.llvm_expr_codegen(shared_from_this());
	}

	const llvm_expr &m_left, &m_right;
};

class llvm_expr_mul : public llvm_expr, public std::enable_shared_from_this<llvm_expr_mul>
{
public:
	llvm_expr_mul(llvm_expr *left, llvm_expr *right, const string &name, uint32_t id, const SPIRType *spir_type)
	    : llvm_expr(name, id, spir_type)
	    , m_left(*left)
	    , m_right(*right)
	{
		assert(m_left.m_spir_type.basetype == m_right.m_spir_type.basetype);
	}

	llvm::Value *codegen(llvm_expr_codegenerator &generator) override
	{
		return generator.llvm_expr_codegen(shared_from_this());
	}

	const llvm_expr &m_left, &m_right;
};

class llvm_expr_div : public llvm_expr, public std::enable_shared_from_this<llvm_expr_div>
{
public:
	llvm_expr_div(llvm_expr *left, llvm_expr *right, const string &name, uint32_t id, const SPIRType *spir_type)
	    : llvm_expr(name, id, spir_type)
	    , m_left(*left)
	    , m_right(*right)
	{
		assert(m_left.m_spir_type.basetype == m_right.m_spir_type.basetype);
	}

	llvm::Value *codegen(llvm_expr_codegenerator &generator) override
	{
		return generator.llvm_expr_codegen(shared_from_this());
	}

	const llvm_expr &m_left, &m_right;
};

} // namespace SPIRV_CROSS_NAMESPACE

#endif /* SPIRV_CROSS_LLVM_LLVM13_HPP */
