# UT C++ — gtest libstdc++ ABI Mismatch

## TL;DR

If `cmake --build tests/ut/cpp/build` fails with `undefined reference to
testing::internal::EqFailure(... std::string const& ...)`, the system
`libgtest.a` that `find_library` picked up was hand-compiled with
`-D_GLIBCXX_USE_CXX11_ABI=0` (the old pre-cxx11 ABI). The tests now
build with the compiler default (cxx11) ABI, so symbol mangling does
not match. Either rebuild gtest without the ABI flag, or force
FetchContent by hiding the system copy:

```bash
cmake -B tests/ut/cpp/build -S tests/ut/cpp \
      -DGTEST_LIB=GTEST_LIB-NOTFOUND -DGTEST_MAIN_LIB=GTEST_MAIN_LIB-NOTFOUND
```

This doc exists so the next person who hits the symptom does not
re-investigate the same rabbit hole.

## Symptom

```text
/usr/bin/ld: CMakeFiles/test_tensormap.dir/hierarchical/test_tensormap.cpp.o:
in function testing::internal::CmpHelperEQFailure<int,int>(...):
undefined reference to `testing::internal::EqFailure(char const*, char const*,
                       std::string const&, std::string const&, bool)`
collect2: error: ld returned 1 exit status
```

The mangled form of the missing symbol is `_ZN7testing8internal9EqFailureEPKcS2_RKSsS4_b`
— `RKSs` is `const std::string&` in the *old* ABI (the new ABI mangles
`std::string` as `NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE`).

Equivalently, `nm /usr/local/lib*/libgtest.a | grep -c __cxx11` returns
`0` on the broken install, ≥1 on a healthy one.

## Root Cause

`tests/ut/cpp/CMakeLists.txt` used to force `-D_GLIBCXX_USE_CXX11_ABI=0`
on every test target and on every runtime OBJECT lib it links. The
matching gtest had to be old-ABI too — fine when FetchContent built it
under the same flag, but a system gtest packaged with the default
(cxx11) ABI link-failed against tests with the symmetric form of the
error.

That `_GLIBCXX_USE_CXX11_ABI=0` was historical baggage:

- `src/` runtime build does **not** set the flag. Production
  `libhost_runtime.so` is already cxx11 — `nm` shows
  `__cxx11::basic_string` mangling on exported C++ symbols (e.g.
  `DeviceRunner::init_pmu`).
- CANN libs (`libascendcl`, `libruntime`, `libhcom`) export only
  `extern "C"` symbols, so the ABI flag is irrelevant at the `.so`
  boundary.

The flag was removed in PR #889 and everything (including the tests
that dlopen the runtime `.so`) is now cxx11 end-to-end. The flip side
is that an old-ABI system gtest, which used to be the "fast" path, now
fails the link. apt / yum / brew packages have all shipped cxx11 ABI
for years; the only way to land in this state is a hand-compiled
gtest left over in `/usr/local`.

## Fixes

Pick one:

1. **Rebuild gtest without the ABI flag.** If you cloned googletest
   and configured it with `cmake -D_GLIBCXX_USE_CXX11_ABI=0` (or with
   an old `simpler` checkout that did so transitively), redo the
   configure without that flag and `make install`.

2. **Drop the manually-installed copy**, let `find_library` miss, and
   FetchContent will rebuild a matching one in the worktree:

   ```bash
   sudo rm /usr/local/lib*/libgtest*.a /usr/local/lib*/libgtest_main*.a
   ```

3. **One-off override**, no global change:

   ```bash
   cmake -B tests/ut/cpp/build -S tests/ut/cpp \
         -DGTEST_LIB=GTEST_LIB-NOTFOUND \
         -DGTEST_MAIN_LIB=GTEST_MAIN_LIB-NOTFOUND
   ```

   Repeat once; the cache is fresh, FetchContent takes over.

## History

- PR #882 added a `try_compile` ABI probe as a *defensive* check while
  the `_GLIBCXX_USE_CXX11_ABI=0` flag was still in place: it caught a
  symmetric mismatch the other way (test=old-ABI vs. a5 self-hosted
  runner's system gtest=new-ABI). That probe is gone now that the
  flag is gone — the whole thing reduces to "use what `find_library`
  finds; fall back to FetchContent if it finds nothing."
- PR #889 dropped the 13 `-D_GLIBCXX_USE_CXX11_ABI=0` sites and the
  probe; this doc records the trail.
