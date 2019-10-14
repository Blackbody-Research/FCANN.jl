using Coverage

function process_folder2(folder="src")
    @info "Coverage.process_folder: Searching $folder for .jl files..."
    source_files = FileCoverage[]
    files = readdir(folder)
    for file in files
        fullfile = joinpath(folder, file)
        if isfile(fullfile)
            # Is it a Julia file?
            if splitext(fullfile)[2] == ".jl"
                push!(source_files, process_file(fullfile, folder))
            else
                @debug "Coverage.process_folder: Skipping $file, not a .jl file"
            end
        elseif isdir(fullfile) && (fullfile != "src")
            # If it is a folder, recursively traverse
            append!(source_files, process_folder2(fullfile))
        end
    end
    return source_files
end