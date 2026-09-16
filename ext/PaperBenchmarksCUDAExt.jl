module PaperBenchmarksCUDAExt

using CUDA
using DFTK
using PaperBenchmarks

function PaperBenchmarks.cuda_architecture()
    DFTK.GPU(CuArray)
end

function PaperBenchmarks.reclaim_memory(::DFTK.GPU{CUDA.CuArray})
    GC.gc(true)
    CUDA.reclaim()
end

end
