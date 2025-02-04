import xml.etree.ElementTree as ET
import sys

if len(sys.argv) != 2 :
    print("Error, run " + sys.argv[0] + ' XML_file')
    exit(1)

mytree = ET.parse(sys.argv[1])
myroot = mytree.getroot()

res = myroot.find('Results')

# Create CSV header
print('Kernel name,Computation duration (us),Global size,Local size', end = ',')
for kr in res.findall('KernelResult') :
    if kr.attrib['Status'] == 'Ok' :
        conf = kr.find('Configuration')
        for tp in conf.findall('Pair') :
            print(tp.attrib['Name'], end = ',')
        crs = kr.find('ComputationResults')
        cr = crs.find('ComputationResult')
        pd = cr.find('ProfilingData')
        if pd != None :
            cts = pd.find('Counters')
            for ct in cts.findall('Counter') :
                print(ct.attrib['Name'], end = ',')
        break
print('Maximum work-group size,Local memory size,Private memory size,Constant memory size,Registers count', end = '')
if 'PowerUsage' in cr.attrib :
    print(',Power,Energy')
else :
    print('')

# Extract data into CSV
for kr in res.findall('KernelResult') :
    if kr.attrib['Status'] == 'Ok' :
        print(kr.attrib['KernelName'], end = ',')
        print(kr.attrib['TotalDuration'], end = ',')

        crs = kr.find('ComputationResults')
        cr = crs.find('ComputationResult')
        gs = cr.find('GlobalSize')
        ls = cr.find('LocalSize')
        gsi = int(gs.attrib['X']) * int(gs.attrib['Y']) * int(gs.attrib['Z'])
        lsi = int(ls.attrib['X']) * int(ls.attrib['Y']) * int(ls.attrib['Z'])
        print(gsi*lsi, end = ',') #TODO manage OpenCL/CUDA format!
        print(lsi, end = ',')

        conf = kr.find('Configuration')
        for tp in conf.findall('Pair') :
            print(tp.attrib['Value'], end = ',')

        pd = cr.find('ProfilingData')
        if pd != None :
            cts = pd.find('Counters')
            for ct in cts.findall('Counter') :
                print(ct.attrib['Value'], end = ',')

        cd = cr.find('CompilationData')
        print(cd.attrib['MaxWorkGroupSize'] + ',' + cd.attrib['LocalMemorySize'] + ',' + cd.attrib['PrivateMemorySize'] + ',' + cd.attrib['ConstantMemorySize'] + ',' + cd.attrib['RegistersCount'], end = '')
        if 'PowerUsage' in cr.attrib :
            print(',' + cr.attrib['PowerUsage'] + ',' + vr.attrib['EnergyConsumption'])
        else :
            print('') #add newline
