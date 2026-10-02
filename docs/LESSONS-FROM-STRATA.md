# What the Strata CUDA -> SYCL port taught us that applies here

The Strata port (an LLM engine, the same card) took a CUDA code base to a SYCL one that beat llama.cpp's SYCL
backend on both prompt and decode speed. These are the lessons that transfer; each cost real time to learn.

## Porting method

- **Port inside upstream's structure, in one directory, and touch no shared file.** Upstream merges then never
  conflict; Intel-only behaviour wraps upstream from outside. For H3 that means `comfy_kitchen/backends/sycl`
  plus a registry entry, nothing in ComfyUI.
- **dpct gets a CUDA tree to "compiles" quickly; its mistakes are silent.** Found only by parity tests:
  `__fadd_rn(a, b)` became `a + b` without parentheses (wrong value inside a ternary); `cudaMemcpy`'s implicit
  stream sync is lost (unwaited copies read reused buffers); device-attribute queries are left as CUDA; inline
  PTX (`cp.async`, `mma.sync`, timers) is left as is. Every kernel gets a parity test before it is trusted.
- **Re-migrate, do not hand-merge.** For an upstream update: dpct the old and the new upstream tree, normalize,
  canonicalize dpct's kernel-name hashes, 3-way merge only the files upstream changed. An audit of the port's
  own identifiers before and after catches dropped code.
- **A fix that recurs belongs in a fixup script**, not in memory.

## Xe2 kernel facts

- **Unaligned 16-byte loads are slow.** Quantized blocks at 2-byte alignment cost a large factor until loads
  were made aligned (second chunk only when unaligned, never reading past the page).
- **Type punning through non-char pointers is not honoured by the device compiler.** Reading an `int2` through
  a `uint16_t*` gave garbage on one format for weeks; extract by shifts.
- **Sub-group size is part of the kernel.** 32 for the warp-style kernels, 16 for XMX `joint_matrix`. The
  ahead-of-time build needs `-fsycl-default-sub-group-size` and the correctly-rounded divide/sqrt option inside
  the backend option string, or results change.
- **Build ahead of time** for the device (`spir64_gen`, `-device bmg-g31`): JIT costs tens of seconds at start
  and the persistent JIT cache has segfaulted on this card.
- **Kernel launches are not free.** Fewer, larger launches beat many small ones; a launch sized for work that
  usually is not there (one block row per possible group) should stride instead.
- **`joint_matrix` XMX did not automatically win.** Our XMX prompt attention was correct and slower than the
  kernel it replaced. Measure before building on it; oneDNN's JIT is the other route to the matrix engine.
- **Lookup tables in registers, not memory**, where they are small (a 16-entry codebook as constants).

## Runtime facts

- **The Level Zero v2 adapter:** an event a host thread has waited on can leave another queue's barrier on the
  same event waiting forever. One queue per pipeline, no cross-queue event waits, completion signalled through
  page-locked memory if a thread must know.
- **Spin waits on the device need a bound** and a way to report giving up.
- **Never kill a process mid-kernel**, and never `docker rm -f` live GPU work: the card can stay wedged until
  a reboot. Stop gracefully, with a timeout.
- **The `xe` driver has no out-of-memory.** Over-allocation evicts VRAM into unswappable host RAM and the host
  livelocks. Cap the allocator, keep 1.5 GB of VRAM free, compute a buffer's size before allocating it.
- **Never build while an engine runs** on a 23 GiB host (8 compile jobs plus a model is an OOM and a watchdog
  reboot).

## Loading

- **The serial load was the slow part of every start**: read one tensor, copy it, wait. Reading in file order
  on several threads into page-locked batches while the previous batch copies moved 23 GiB in 19 s instead of
  76 s. H3's 21 GB denoiser is loaded lazily inside the first step today.
- **Pinned host memory** is what makes host-to-card copies run at the link's speed (6.5 GB/s on PCIe 3.0 x8).

## Measuring

- **Compare like with like.** Outputs that differ change downstream work (draft acceptance there, nothing
  here - but a changed checkpoint changes the picture), so compare time per unit of identical work.
- **A one-off stall can land in either of two timers**; sum them or run long enough.
- **A "hang" with a busy GPU and a quiet log** was twice a deadlock and once just a silent, slow path. A debug
  switch that syncs and logs after every phase tells them apart in one run.
- **Keep the numbers in the document current**: when a speed moves, re-run every table that quotes it.
