import requests
import os
import time


patterns = {
              '2015-01':['yellow', 'green', 'fhv'],
              '2015-02':['yellow', 'green', 'fhv'],
              '2015-03':['yellow', 'green', 'fhv'],
              '2015-04':['yellow', 'green', 'fhv'],
              '2015-05':['yellow', 'green', 'fhv'],
              '2015-06':['yellow', 'green', 'fhv'],
              '2015-07':['yellow', 'green', 'fhv'],
              '2015-08':['yellow', 'green', 'fhv'],
              '2015-09':['yellow', 'green', 'fhv'],
              '2015-10':['yellow', 'green', 'fhv'],
              '2015-11':['yellow', 'green', 'fhv'],
              '2015-12':['yellow', 'green', 'fhv'],
              '2014-01':['yellow', 'green'],
              '2014-02':['yellow', 'green'],
              '2014-03':['yellow', 'green'],
              '2014-04':['yellow', 'green'],
              '2014-05':['yellow', 'green'],
              '2014-06':['yellow', 'green'],
              '2014-07':['yellow', 'green'],
              '2014-08':['yellow', 'green'],
              '2014-09':['yellow', 'green'],
              '2014-10':['yellow', 'green'],
              '2014-11':['yellow', 'green'],
              '2014-12':['yellow', 'green'],        
              '2013-01':['yellow'],
              '2013-02':['yellow'],
              '2013-03':['yellow'],
              '2013-04':['yellow'],
              '2013-05':['yellow'],
              '2013-06':['yellow'],
              '2013-07':['yellow'],
              '2013-08':['yellow', 'green'],
              '2013-09':['yellow', 'green'],
              '2013-10':['yellow', 'green'],
              '2013-11':['yellow', 'green'],
              '2013-12':['yellow', 'green'],
              '2012-01':['yellow'],
              '2012-02':['yellow'],
              '2012-03':['yellow'],
              '2012-04':['yellow'],
              '2012-05':['yellow'],
              '2012-06':['yellow'],
              '2012-07':['yellow'],
              '2012-08':['yellow'],
              '2012-09':['yellow'],
              '2012-10':['yellow'],
              '2012-11':['yellow'],
              '2012-12':['yellow'],
              '2011-01':['yellow'],
              '2011-02':['yellow'],
              '2011-03':['yellow'],
              '2011-04':['yellow'],
              '2011-05':['yellow'],
              '2011-06':['yellow'],
              '2011-07':['yellow'],
              '2011-08':['yellow'],
              '2011-09':['yellow'],
              '2011-10':['yellow'],
              '2011-11':['yellow'],
              '2011-12':['yellow'],
              '2010-01':['yellow'],
              '2010-02':['yellow'],
              '2010-03':['yellow'],
              '2010-04':['yellow'],
              '2010-05':['yellow'],
              '2010-06':['yellow'],
              '2010-07':['yellow'],
              '2010-08':['yellow'],
              '2010-09':['yellow'],
              '2010-10':['yellow'],
              '2010-11':['yellow'],
              '2010-12':['yellow'],
              '2009-01':['yellow'],
              '2009-02':['yellow'],
              '2009-03':['yellow'],
              '2009-04':['yellow'],
              '2009-05':['yellow'],
              '2009-06':['yellow'],
              '2009-07':['yellow'],
              '2009-08':['yellow'],
              '2009-09':['yellow'],
              '2009-10':['yellow'],
              '2009-11':['yellow'],
              '2009-12':['yellow'],
              }


dates = [
       '2015-01','2011-10', '2012-12', '2014-02', '2010-03', 
       '2012-07','2014-10', '2014-06', '2011-12', '2013-06', 
       '2009-02', '2010-08','2012-08', '2013-11', '2011-06', 
       '2011-03', '2011-08', '2015-02','2013-01', '2014-09', 
       '2013-04', '2013-09', '2011-05', '2011-11','2010-11', 
       '2010-05', '2012-06', '2013-03', '2010-09', '2013-07',
       '2009-12', '2009-10', '2011-09', '2015-06', '2014-07', 
       '2009-06','2012-09', '2009-03', '2009-08', '2009-05', 
       '2011-02', '2015-05','2013-02', '2015-04', '2010-12', 
       '2014-04', '2014-08', '2012-01','2009-09', '2014-03', 
       '2012-05', '2013-12', '2011-07', '2009-01','2012-11', 
       '2014-05', '2011-04', '2014-11', '2009-11', '2012-02',
       '2013-08', '2012-10', '2013-10', '2010-02', '2014-12', 
       '2011-01','2010-10', '2012-04', '2012-03', '2010-01', 
       '2014-01', '2010-07','2009-04', '2013-05', '2010-04', 
       '2010-06', '2009-07', '2015-03'
       ]



def write_part(file, part):
       with open(file, 'ab+') as f:
              f.write(part)


def parce(link):
       date = link[1]['date']
       pattern = link[1]['pattern']
       storage_content = os.listdir('/Volumes/Sokol76/')
       file_name = '/Volumes/Sokol76/taxi_{}_{}.csv'.format(date, pattern)
       if file_name.split('/')[-1] in storage_content:
              return None

       url = link[0]
       response = requests.get(url)
       size = len(response.content)
       write_part(file_name, response.content[:size//2])
       write_part(file_name, response.content[size//2:])
       return None


def link(pattern, date):
       l = 'https://s3.amazonaws.com/nyc-tlc/trip+data/{0}_tripdata_{1}.csv'.format(pattern, date)
       return l


if __name__ == '__main__':

       links = []
       for date in dates:
              if len(patterns[date]) == 1:
                     pattern = patterns[date][0]
                     tempelate = link(pattern, date)
                     links.append([tempelate,{'date': date, 'pattern': pattern}])
              else:
                     for i in range(len(patterns[date])):
                            pattern = patterns[date][i]
                            tempelate = link(pattern, date)
                            links.append([tempelate,{'date': date, 'pattern': pattern}])

       for link in links:
              try:
                     parce(link)
                     t = time.localtime()
                     print('Done at {0}:{1}'.format(t.tm_hour, t.tm_min), link[1])
              except OSError:
                     print('****NO', link)
                     break
                     