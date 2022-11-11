module Models

using MatrixProductBP

export ...

include("glauber/glauber.jl")
include("glauber/bp_glauber.jl")
include("sis/sis.jl")
include("sis/bp_sis.jl")

end # end module