"""
Process simulation results and arrange a training data for Social GAN
"""
using JDL

path_dir_sim_results = "../sim_records"
files = readdir(path_dir_sim_results)

# TODO: 

for f in files
    vars = load(join([path_dir_sim_results, f],"/"))
    for n in vars.rec.nframes
    end
end
