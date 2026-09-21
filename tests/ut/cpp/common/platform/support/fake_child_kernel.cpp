/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * A loadable stand-in for a compiled AICore child kernel.
 *
 * The sim registration path writes each child's binary to a temp file, dlopens
 * it and resolves `kernel_entry`, so a test that wants that path to succeed
 * needs bytes that really are a shared object exporting that symbol. This is
 * built as a module beside the tests and read back as data; nothing calls it.
 */

extern "C" void kernel_entry(void * /*args*/) {}
