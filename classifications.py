"""
This program contains data for classifying drugs by their effects.

Project: EcstasyData
Path: root/classifications.py
"""
# MDMA
MDMA = ['MDMA']

# Enactogens
ecstasy_like = ['MDA', 'MDE', '4-FA', 'Methylone', 'bk-DMBDB', 'bk-MDDMA', 'Ethylone', '4-Methylmethcathinone', \
	'bk-MBDB', '5-MAPB', '6-APB', '5-APDB', '5-EAPB', '5-APB', 'AMT', '3-Methylmethcathinone', '4-Ethylmethcathinone', \
	'5- or 6-MAPB', '5-MAPB', '6-MAPB']

# Psychedelics
psychedelics_other = ['PMA', 'Bromo-dragonfly', 'AL-LAD', 'Psilocin', '4-HO-MET', \
	'25C-NBOMe', '25I-NBOMe', 'LSD', '1-(2-fluorophenyl)piperazine', 'Mescaline', 'Escaline', 'Proscaline']
family_2C = ['2C-B', '2C-BAN', '2C-Bn', '2C-Bu', '2C-C', '2C-CN', '2C-CP', '2C-D', '2C-E', '2C-EF', '2C-F', '2C-G', '2C-G-1', \
	'2C-G-2', '2C-G-3', '2C-G-4', '2C-G-5', '2C-G-6', '2C-G-N', '2C-H', '2C-I', '2C-iP', '2C-N', '2C-NH2', '2C-O', '2C-O-4', \
	'2C-P', '2C-Ph', '2C-SE', '2C-T', '2C-T-2', '2C-T-3', '2C-T-4', '2C-T-5', '2C-T-6', '2C-T-7', '2C-T-8', '2C-T-9', '2C-T-10', \
	'2C-T-11', '2C-T-12', '2C-T-13', '2C-T-14', '2C-T-15', '2C-T-16', '2C-T-17', '2C-T-18', '2C-T-19', '2C-T-20', '2C-T-21', \
	'2C-T-21.5', '2C-T-22', '2C-T-23', '2C-T-24', '2C-T-25', '2C-T-27', '2C-T-28', '2C-T-30', '2C-T-31', '2C-T-32', '2C-T-33', \
	'2C-TFE', '2C-TFM', '2C-YN', '2C-V', \
	'25H-NBOMe', '25B-NBOMe', '25B-NBOH', '25I-NBOH', '25C-NBOMe']
family_psychedelic_amphetamine = ['DOC', 'DOB']
family_tryptamine = ['DMT', '5-MeO-DMT', '5-MeO-MiPT', '5-MeO-DiPT', '5-MeO-DPT', '5-MeO-DALT', 'DPT', '5-MeO-MIPT', '4-Acetoxy-DMT', '4-HO-MIPT']
psychedelics = psychedelics_other + family_2C + family_tryptamine + family_psychedelic_amphetamine

# Cannabinoids
cannabinoids = ['AB-PINACA', 'JWH-018', 'JWH-073', 'JWH-081', 'JWH-210', 'JWH-250', 'JWH-359', 'AB-CHMINACA', 'AB-FUBINACA', 'THC', 'Thermoliz UR-144F', 'UR-144', \
'5-fluoro-AMB', '5-fluoro-ADB ', 'ADB-CHMINACA', 'ADB-FUBINACA', 'AMB-FUBINACA', 'NM-2201', 'U-47700', 'XLR-11']

# Dissociatives
dissociatives = ['DXM', 'Ketamine', 'PCP', 'Methoxetamine', 'Deschloroketamine', '3-MeO-PCP', '4-MeO-PCP', 'Methoxphenidine', '2-fluoro-Deschloroketamine', '2\'-Oxo-PCE']

# Stimulants
stimulants = ['Amphetamines', 'BZP', 'Caffeine', 'Cocaine', 'Methamphetamine', 'Pseudo/Ephedrine', 'TFMPP', 'MDPV', 'Dimethylcathinone', 'MDAI', 'Ethylamphetamine', \
	'Dibenzylpiperazine', 'mCPP', 'Modafinil', 'Phentermine', 'N-Ethylpentylone', 'Fluoroamphetamine (2,3, or 4 fluoro)', 'Diethylpropion', '1,1-diphenylprolinol', \
	'4-CEC', '4-CMC', 'alpha-PVP', 'Tropacocaine', '3-Fluorophenmetrazine', 'Nicotine', 'Methylphenidate', '4-Fluoromethylphenidate', 'N-Ethylhexedrone', 'MBZP', \
	'4-MEC', '4-Fluoromethcathinone', 'alpha-PHP', 'Isopropylphenidate', 'Ethylphenidate', 'Pentylone', 'Methiopropamine', '4-fluoro-alpha-PHP', 'Pentedrone', \
	'2-FMA', 'a-PBP', '4-fluoro-alpha-PVP', 'Norcocaine']

# Depressants/Tranquilizers
opioids = ['Codeine', 'Opium', 'Heroin', 'Tramadol', 'Oxycodone', 'Dihydrocodeine', 'Hydrocodone', 'Acetylcodeine', \
	'Butyrfentanyl', 'Fentanyl', 'Furanylfentanyl', 'Acetylfentanyl', 'Benzyl fentanyl', 'Despropionyl fentanyl']
benzodiazepines = ['Lorazepam', 'Diazepam', 'Etizolam', 'Alprazolam', 'Clonazolam', 'Clonazepam']
barbiturates = ['Butabarbital', 'Phenobarbital', 'Barbital', 'Pentobarbital', 'Butalbital']
depressants = ['Zolpidem', 'Amitriptyline',  'Carisoprodol', 'Diphenhydramine', 'Methaqualone'] + opioids + benzodiazepines + barbiturates

# Other
research_chemicals = ['4-Chloro-N,N-Dimethylcathinone']
precursors_intermediates_byproducts = ['MMDPPA', 'MDA 2-amido analog', '1-(3,4-methylenedioxyphenyl)-2-propanol', '4-ANPP', 'MDP2P']
no_effect = ['Acetaminophen', 'Sugar', 'Aspirin', 'Methyl Salicylate', 'PMMA', 'Ibuprofen', 'Vitamin E', 'Oleic Acid', 'Palmitic Acid', \
	'Stearic Acid', 'Linoleic acid', 'Finasteride', 'Phthalates', 'Methylecgonidine', 'Triethyl citrate']
steroids = ['Mesterolone', 'Methandrostenolone', 'Oxandrolone', 'Fluoxymesterone']
nootropics = ['Piracetam']
misc_others = ['Chlorpheniramine', 'Guaifenesin', 'Phenylpropanolamine', 'Niacinamide', \
	'Methylsulfonylmethane', 'Procaine', 'Phenacetin', 'Lidocaine', 'Benzocaine', 'Novocaine', 'Fluoxetine', 'Allopurinol', 'Levamisole', \
	'Benocyclidine', 'MDA 2-aldoxime analog', 'MDMA Methylene homolog', 'Melatonin', 'Sildenafil', 'Tadalafil', \
	'Stanozolol', 'Sibutramine', 'Diclofenac', 'Aceclofenac', 'W-15', 'Theophylline', \
	'Unidentified', 'Other Pharm.', 'Synthesis Byproducts', 'Other', 'None detected', 'Trace detected']
silent_others = no_effect + misc_others + steroids + nootropics + precursors_intermediates_byproducts + research_chemicals

aliases_for_nothing = ['Not tested', 'Not Tested', '', 'Not tested:', 'Not Tested:', 'Not tested:---', 'Not Tested:---', \
	'Not tested:1', 'Not Tested:1', 'On Hold - See Note:0', 'On Hold - See Note:', 'On Hold - See Note:1']
	