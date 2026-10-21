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

function PaperBenchmarks._seed_rng!(::DFTK.GPU{CUDA.CuArray}, seed::Integer)
    CUDA.seed!(seed)
end

end
