module PaperBenchmarksCUDAExt

using CUDA
using DFTK
using PaperBenchmarks

function PaperBenchmarks.cuda_architecture()
    DFTK.GPU(CuArray)
end

end
