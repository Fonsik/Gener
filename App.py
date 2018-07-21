#!/usr/bin/env python
import ROOT
from ROOT import TMVA, TFile, TString, TTree, TChain
from array import array
from subprocess import call
from os.path import isfile
import array
# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
reader = TMVA.Reader("Color:!Silent")
data = ROOT.TChain("dat","")
data.Add("./data.root/dat")

os = ROOT.TFile("output_selection.root","RECREATE")
sel = TTree( 'Selection', 'Selection outputs' )

branches = {}
for branch in data.GetListOfBranches():
    branchName = branch.GetName()
    if (branchName!='a' and branchName!='b'):
        branches[branchName] = array.array('f', [0])
        reader.AddVariable(branchName, branches[branchName])
        sel.Branch(branchName, branches[branchName], branchName+"/F")

# Book methods
reader.BookMVA('PyKeras', TString('dataset/weights/TMVAClassification_PyKeras.weights.xml'))

# Print some example classifications
print reader.EvaluateMVA('PyKeras')
print('Some signal example classifications:')
a=data.GetEntries()
for i in range(20):
    data.GetEntry(i)
    ev=reader.EvaluateMVA('PyKeras')
    print(ev)
    print data.a00
print('')

'''

#!/usr/bin/env python

from ROOT import TMVA, TFile, TString
from array import array
from subprocess import call
from os.path import isfile

# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
reader = TMVA.Reader("Color:!Silent")

# Load data
if not isfile('tmva_reg_example.root'):
    call(['curl', '-O', 'http://root.cern.ch/files/tmva_reg_example.root'])

data = TFile.Open('tmva_reg_example.root')
tree = data.Get('TreeR')

branches = {}
for branch in tree.GetListOfBranches():
    branchName = branch.GetName()
    branches[branchName] = array('f', [-999])
    tree.SetBranchAddress(branchName, branches[branchName])
    if branchName != 'fvalue':
        reader.AddVariable(branchName, branches[branchName])

# Book methods
reader.BookMVA('PyKeras', TString('dataset/weights/TMVARegression_PyKeras.weights.xml'))

# Print some example regressions
print('Some example regressions:')
for i in range(20):
    tree.GetEntry(i)
    print('True/MVA value: {}/{}'.format(branches['fvalue'][0],reader.EvaluateMVA('PyKeras')))

'''