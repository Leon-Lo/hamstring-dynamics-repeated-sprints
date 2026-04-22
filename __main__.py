def main():
    print(
        "Hamstring Length During Repeated Sprints\n"
        "\n"
        "Usage:\n"
        "  conda activate opensim\n"
        "\n"
        "  # Run analyses (requires data/master_results.csv)\n"
        "  python analysis/participant_demographics.py\n"
        "  python analysis/sprint_sp_t_analysis.py\n"
        "  python analysis/fatigue_analysis.py\n"
        "  python analysis/peak_fiber_length_analysis.py\n"
        "  python analysis/responder_analysis.py\n"
        "  python analysis/supplementary_analysis.py\n"
        "  python analysis/publication_figures.py\n"
        "\n"
        "  # Re-run simulations from raw data (requires OpenSim models + CEINMS)\n"
        "  python analysis/data_processing.py\n"
        "\n"
        "See README.md for full setup and data structure instructions."
    )


if __name__ == "__main__":
    main()
