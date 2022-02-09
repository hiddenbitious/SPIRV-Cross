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

#ifndef SPIRV_CROSS_LLVM_HPP
#define SPIRV_CROSS_LLVM_HPP

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "spirv_glsl.hpp"

namespace SPIRV_CROSS_NAMESPACE
{

class CompilerLLVM : public CompilerGLSL
{
public:
	CompilerLLVM(std::vector<uint32_t> spirv_);
	CompilerLLVM(const uint32_t *ir_, size_t word_count);
	explicit CompilerLLVM(const ParsedIR &ir_);
	explicit CompilerLLVM(ParsedIR &&ir_);
	~CompilerLLVM();

	std::string compile() override;

	const void *get_entry_point_fp()
	{
		return m_bin_fp;
	}

	spv::ExecutionModel get_execution_model(void) const
	{
		return m_spirv_entry_point.model;
	}

	uint32_t get_number_of_inputs(void) const
	{
		return m_n_inputs;
	}

	uint32_t get_number_of_outputs(void) const
	{
		return m_n_outputs;
	}

	struct Options
	{
		bool optimize_llvm = false;
	};

	const Options &get_llvm_options() const
	{
		return m_llvm_options;
	}

	void set_llvm_options(const Options &opts)
	{
		m_llvm_options = opts;
	}

	class CompilerLLVM_impl;
	CompilerLLVM_impl *m_pimpl = nullptr;

private:
	void emit_header() override;
	void emit_function_prototype(SPIRFunction &func, const Bitset &return_flags) override;
	void emit_function(SPIRFunction &func, const Bitset &return_flags);
	void emit_block_chain(SPIRBlock &block);
	void emit_block_instructions(SPIRBlock &block);
	void emit_instruction(const Instruction &instruction) override;
	void emit_resources();
	void emit_interface_block(const SPIRVariable &var);

	ID get_next_id(void)
	{
		m_current_id = ID(m_current_id + 1);

		return m_current_id;
	}

	void emit_default_shader_variables();
	void emit_default_vertex_shader_variables();
	void emit_default_fragment_shader_variables();
	void emit_glsl_function_prototypes();

	// Creates a list of the implicit entry point arguments introduced and used internally by the implementation.
	// The arguments vary depending on the shader stage.
	void construct_implicit_entry_point_arguments(std::vector<SPIRType> &arg_types, std::vector<std::string> &arg_names,
	                                              std::vector<uint32_t> &arg_ids);

	// Creates a list of the implicit entry point arguments introduced and used internally by the implementation for the vertex shader.
	// - vec4<float> *_attribs_			An array of vec4<float> holding the value for each vertex attribute.
	// - vec4<float> *__gl_Position		A pointer to a vec4<float> to write the clip coordinates and varying values.
	void construct_implicit_vertex_entry_point_arguments(std::vector<SPIRType> &arg_types,
	                                                     std::vector<std::string> &arg_names,
	                                                     std::vector<uint32_t> &arg_ids);

	// Creates a list of the implicit entry point arguments introduced and used internally by the implementation for the fragment shader.
	// - vec4<float> *__varyings_in				An array of vec4<float> holding the value for each varying.
	// - vec4<float> *__descriptor_table__		An array of void *. Each pointer points to a descriptor set.
	// - vec4<float> *__color_attachment_0		A pointer to a vec4<float> to the color attachment
	void construct_implicit_fragment_entry_point_arguments(std::vector<SPIRType> &arg_types,
	                                                       std::vector<std::string> &arg_names,
	                                                       std::vector<uint32_t> &arg_ids);

	// Loads the shader stage's input variables from the implicit entry point function's arguments
	void emit_entry_point_input_loads(void);

	// Stores the shader's output variables into the entry point's implicit arguments
	void emit_shader_outputs_stores(void);

	// Stores the vertex shader's output variables into the entry point's implicit arguments.
	void emit_vertex_shader_varyings_store(void);

	void emit_fragment_shader_color_attachment_stores(void);

	uint32_t m_n_inputs;
	uint32_t m_n_outputs;

	// Keep track of current ID. Starts counting from the upper bound ID specified in the spirv header.
	ID m_current_id;

	Options m_llvm_options;

	// Stage's entry point function
	SPIREntryPoint m_spirv_entry_point;

	void *m_bin_fp;
};

} // namespace SPIRV_CROSS_NAMESPACE

#endif
