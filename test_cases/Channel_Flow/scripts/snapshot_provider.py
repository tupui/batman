"""Example of snapshots providers."""

job_llsubmit = {
# Command line or python function to invoque for computing a snapshot.
# If it's a shell command line, it will be executed from the context directory (see 'context' key).
# format : python function object or string ('bash', 'qsub', 'sbatch')
    'command' : 'bash', 
# Script to be executed by the command
# The script path must be relative to the context directory.
# format : string
    'script' : 'data/generator_llsubmit/script.sh',
# Context for computing a snapshot
# Path to the directory that contains the files needed to run a snapshot computation. It is used as a template, the actual computation will not be ran from this location.
    'context' : 'data/generator_llsubmit',
# JPOD private directory
# Name of a directory where jpod will put some files while the snapshot production is going on. This directory will be located in the snapshot producer working directory.
    'private-directory' : 'jpod-data',
# Location of snapshot files
# Relative path of a directory where jpod will read the snapshot files.
    'data-directory' : 'cfd-output-data',
# Task completion timeout
# Maximum duration of a snapshot task in seconds
    'timeout' : 3600,
# Restart a process if it fails.
# format : 'True' or 'False'
    'restart'  : 'False',
    'clean'  : False,
}
