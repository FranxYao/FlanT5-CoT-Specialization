from datetime import datetime
import pytz 

def tprint(str):
  timezone = pytz.timezone("America/Vancouver")
  timenow = datetime.now(timezone)
  currenttime= timenow.strftime("%m/%d/%Y, %H:%M:%S")
  print(currenttime + ' ' + str)
  return  