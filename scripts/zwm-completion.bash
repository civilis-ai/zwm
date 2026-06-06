# Bash completion for `zwm` — source from your .bashrc:
#
#     eval "$(register-python-argcomplete zwm)"
#
# This file is a fallback for users who don't have
# `argcomplete` installed; it provides a minimal but correct
# completion spec covering all sub-commands introduced in
# P4 (config unification) and H1-H3 (production hardening).

_zwm_completion() {
    local cur prev cmds
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    cmds="tick eval replay info inspect serve mcp mcp-http otlp spans a2a a2a-serve"

    if [[ ${COMP_CWORD} -eq 1 ]] ; then
        COMPREPLY=( $(compgen -W "${cmds}" -- "${cur}") )
        return 0
    fi

    case "${COMP_WORDS[1]}" in
        tick|eval)
            COMPREPLY=( $(compgen -W "--steps --year --seed --period --mcts_iterations --n_particles --use_diffusion --learnable_encoder --hierarchical --use_fsdp2 --use_react --db_path --json" -- "${cur}") )
            ;;
        replay|inspect)
            COMPREPLY=( $(compgen -W "--db --limit --show-reflections --json" -- "${cur}") )
            ;;
        info)
            COMPREPLY=( $(compgen -W "--json" -- "${cur}") )
            ;;
        serve|mcp-http|a2a-serve)
            COMPREPLY=( $(compgen -W "--host --port --reload --log-level" -- "${cur}") )
            ;;
        mcp|otlp|spans|a2a)
            COMPREPLY=( $(compgen -W "--help" -- "${cur}") )
            ;;
    esac
    return 0
}

complete -F _zwm_completion zwm
