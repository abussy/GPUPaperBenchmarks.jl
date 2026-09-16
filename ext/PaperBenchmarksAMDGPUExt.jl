module PaperBenchmarksAMDGPUExt

using AMDGPU
using DFTK
using PaperBenchmarks

function PaperBenchmarks.amdgpu_architecture()
    DFTK.GPU(ROCArray)
end

end
