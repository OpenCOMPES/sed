import sys
sys.path.append("/home/zains/sed/")
from sed.dataframe_reader import dataframeReader

prc = dataframeReader('/home/zains/sed/tutorial/config.yml', runNumber=22097)
print(prc.dd)