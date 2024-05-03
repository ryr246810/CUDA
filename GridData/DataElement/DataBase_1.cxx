#include <DataBase.hxx>
#include <GeomDataBase.hxx>


void 
DataBase::
SetupSweptPhysData(Standard_Integer _datanum)
{
  ClearSweptPhysData();
  m_SweptPhysDataNum = _datanum;
  m_SweptPhysData = new Standard_Real[m_SweptPhysDataNum];
  ZeroSweptPhysDatas();
}


void 
DataBase::
ZeroSweptPhysDatas()
{
  for(Standard_Integer index=0; index<m_SweptPhysDataNum; index++){
    m_SweptPhysData[index]=0.0;
  }
}


bool 
DataBase::
IsSweptPhysDataDefined()  const 
{
  bool result = true;
  if(m_SweptPhysData==NULL){
    result = false;
  }
  return result;
}


void 
DataBase::
ClearSweptPhysData()
{
  if(m_SweptPhysData!=NULL){
    delete[] m_SweptPhysData;
  }
  m_SweptPhysData=NULL;
}

 
Standard_Size  
DataBase::
GetSweptPhysDataNum() const
{
  return m_SweptPhysDataNum;
}


Standard_Real* 
DataBase::
GetSweptPhysData()
{
  return m_SweptPhysData;
}


Standard_Real * DataBase::GetSweptPhysDataPtr(Standard_Integer _index){
  return &(m_SweptPhysData[_index]);
}


Standard_Real  
DataBase::
GetSweptPhysData(Standard_Integer _index) const
{
  return m_SweptPhysData[_index];
}


void 
DataBase::
SetSweptPhysData(Standard_Integer _index, Standard_Real _value)
{
  m_SweptPhysData[_index] = _value;
}


void 
DataBase::
AddSweptPhysData(Standard_Integer _index, Standard_Real _value)
{
  m_SweptPhysData[_index] += _value;
}


void 
DataBase::
SubtractSweptPhysData(Standard_Integer _index, Standard_Real _value)
{
  m_SweptPhysData[_index] -= _value;
}

void DataBase::ResetSweptPhysDataPtr(Standard_Integer _datanum, Standard_Real * valueptr)
{
  ClearSweptPhysData();
  m_SweptPhysDataNum = _datanum;
  m_SweptPhysData = valueptr;
  //m_MaterialData = valueptr + sizeof(Standard_Real)*m_SweptPhysDataNum;
  //ZeroPhysDatas();
}

void DataBase::CleanSweptPhysDataPtr()
{
  m_SweptPhysDataNum = 0.0;
  m_SweptPhysData = NULL;
}
