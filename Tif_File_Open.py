import os
import math as m
import numpy as np
from osgeo import gdal
import linecache

projection = [
        # WGS84坐标系(EPSG:4326)
        """GEOGCS["WGS 84", DATUM["WGS_1984", SPHEROID["WGS 84", 6378137, 298.257223563, AUTHORITY["EPSG", "7030"]], AUTHORITY["EPSG", "6326"]], PRIMEM["Greenwich", 0, AUTHORITY["EPSG", "8901"]], UNIT["degree", 0.01745329251994328, AUTHORITY["EPSG", "9122"]], AUTHORITY["EPSG", "4326"]]""",
        # Pseudo-Mercator、球形墨卡托或Web墨卡托(EPSG:3857)
        """PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext  +no_defs"],AUTHORITY["EPSG","3857"]]"""
    ]

def Get_Data(filepath):
      dataset = gdal.Open(filepath)#文件打开
      #获取文件基本信息
      im_width = dataset.RasterXSize #栅格矩阵的列数
      im_height = dataset.RasterYSize #栅格矩阵的行数
      im_bands = dataset.RasterCount #波段数
      for b in range(dataset.RasterCount):
            # 注意GDAL中的band计数是从1开始的
            band = dataset.GetRasterBand(b + 1)
            # 波段数据的一些信息
            print(f'数据类型：{gdal.GetDataTypeName(band.DataType)}')  # DataType属性返回的是数字
            print(f'NoData值：{band.GetNoDataValue()}')  # 很多影像都是NoData，要特别对待
            print(f'统计值（最大值最小值）：{band.ComputeRasterMinMax()}')  # 有些数据本身就存储了统计信息，有些数据没有需要计算  
      #获取数据
      im_data = dataset.ReadAsArray(0,0,im_width,im_height)#将读取的数据作为numpy的Array      
      print('Read Successfully')
      return im_data

def file_geoTransform(filepath):
      dataset=gdal.Open(filepath)
      return dataset.GetGeoTransform()#获取仿射矩阵信息

def file_geoProjection(filepath):
      dataset=gdal.Open(filepath)
      return dataset.GetProjection()#获取图像投影信息 

def get_BIPImage(imgarr):
      #从数据中提取波段(当前仅针对Landsat数据进行处理)
      im_BIPArray=np.append(\
            imgarr[2,0:imgarr.shape[1],0:imgarr.shape[2]].reshape(imgarr.shape[1]*imgarr.shape[2],1),\
            imgarr[1,0:imgarr.shape[1],0:imgarr.shape[2]].reshape(imgarr.shape[1]*imgarr.shape[2],1),axis=1)#合并红绿波段

      im_BIPArray=np.append(im_BIPArray,\
            imgarr[0,0:imgarr.shape[1],0:imgarr.shape[2]].reshape(imgarr.shape[1]*imgarr.shape[2],1),axis=1)#合并红绿蓝波段
      if imgarr.shape[0]>3:
            im_BIPArray=np.append(im_BIPArray,\
            imgarr[3,0:imgarr.shape[1],0:imgarr.shape[2]].reshape(imgarr.shape[1]*imgarr.shape[2],1),axis=1)#合并红绿蓝近红外波段

            im_BIPArray=im_BIPArray.reshape(imgarr.shape[1],imgarr.shape[2],self.im_bands)#调整图像尺寸
            return im_BIPArray


def Write_Data(imgarr, Data_Projection, Data_GeoTransform,save_path):
      driver = gdal.GetDriverByName("GTiff")
      #基本数据信息准备
      img_width = imgarr.shape[2]
      img_height = imgarr.shape[1]
      num_bands = imgarr.shape[0]
      datatype=imgarr.GDT_Float32
      if 'int8' in imgarr.dtype.name:
            datatype = gdal.GDT_Byte
      elif 'int16' in imgarr.dtype.name:
            datatype = gdal.GDT_UInt16
      
      dataset=driver.Create(save_path,img_width, img_height, num_bands, datatype)
      if dataset is not None:
            dataset.SetGeoTransform(Data_GeoTransform)#写入仿射变换参数
            dataset.SetProjection(Data_Projection)#写入投影参数
      for i in range(num_bands):
            dataset.GetRasterBand(i + 1).WriteArray(imgarr[i])

      print("Write Success")



