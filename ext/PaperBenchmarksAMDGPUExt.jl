module PaperBenchmarksAMDGPUExt

using AMDGPU
using DFTK
using PaperBenchmarks

function PaperBenchmarks.amdgpu_architecture()
    DFTK.GPU(ROCArray)
end

function PaperBenchmarks.reclaim_memory(::DFTK.GPU{AMDGPU.ROCArray})
    GC.gc(true)
    AMDGPU.HIP.reclaim()
end

end
