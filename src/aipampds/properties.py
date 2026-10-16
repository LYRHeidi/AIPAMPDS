"""Physicochemical descriptors using the reference platform's conventions."""


def calculate_properties(sequences):
    """Return amidated descriptors at pH 7.4 with Eisenberg hydrophobicity."""
    import pandas as pd
    from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

    global_descriptor = GlobalDescriptor(sequences)
    global_descriptor.calculate_all(amide=True, ph=7.4)
    frame = pd.DataFrame(global_descriptor.descriptor, columns=global_descriptor.featurenames)
    peptide_descriptor = PeptideDescriptor(sequences, "eisenberg")
    peptide_descriptor.calculate_moment()
    frame["Hydrophobic Moment"] = peptide_descriptor.descriptor.reshape(-1)
    peptide_descriptor.calculate_global()
    frame["Hydrophobicity"] = peptide_descriptor.descriptor.reshape(-1)
    return frame.rename(columns={"InstabilityInd": "Instability Index"})[
        ["Length", "MW", "Charge", "Instability Index", "Hydrophobicity", "Hydrophobic Moment"]
    ]
