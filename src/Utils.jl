"""
this returns the path of the home directory in the project
"""
function return_home_dir()::String
    current_dir::String = pwd()
    range::UnitRange{Int64} = findlast(".jl", current_dir)
    isnothing(range) && throw(SystemError("Impossible to set the home dir"))
    return current_dir[1:range[2]+1]
end
