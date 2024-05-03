#include <DataBase.hxx>
#include <GeomDataBase.hxx>

void DataBase::SetupPhysData(Standard_Integer _datanum)
{
  ClearPhysData();

  m_PhysDataNum = _datanum;
  m_PhysData = new Standard_Real[m_PhysDataNum];

  ZeroPhysDatas();
}


void DataBase::ZeroPhysDatas()
{
  for(Standard_Integer index=0; index<m_PhysDataNum; index++){
    m_PhysData[index]=0.0;
  }
}


bool DataBase::IsPhysDataDefined()  const 
{
  bool result = true;
  if(m_PhysData==NULL){
    result = false;
  }
  return result;
}


void DataBase::ClearPhysData()
{
  if(m_PhysData!=NULL){
    delete[] m_PhysData;
  }
  m_PhysData=NULL;
}


Standard_Size DataBase::GetPhysDataNum()  const
{
  return m_PhysDataNum;
}


Standard_Real* DataBase::GetPhysData()
{
  return m_PhysData;
}

Standard_Real DataBase::GetPhysData(Standard_Integer _index)  const
{
  return m_PhysData[_index];
}


Standard_Real * DataBase::GetPhysDataPtr(Standard_Integer _index){
  return &(m_PhysData[_index]);
}


void DataBase::SetPhysData(Standard_Integer _index, Standard_Real _value)
{
  m_PhysData[_index] = _value;
}


void DataBase::AddPhysData(Standard_Integer _index, Standard_Real _value)
{
  m_PhysData[_index] += _value;
  // std:: cout << &(m_PhysData[_index]) << "+++++++++++";
}


void DataBase::SubtractPhysData(Standard_Integer _index, Standard_Real _value)
{
  m_PhysData[_index] -= _value;
}

void DataBase::ResetPhysDataPtr(Standard_Integer _datanum, Standard_Real * valueptr)
{
  ClearPhysData();
  m_PhysDataNum = _datanum;
  m_PhysData = valueptr;
  ZeroPhysDatas();
}

void DataBase::ResetPhysDataPtr(Standard_Integer _datanum, Standard_Real * valueptr, Standard_Integer _isElecEdge)
{
  ClearPhysData();
  m_PhysDataNum = _datanum;
  m_PhysData = valueptr;
  valueptr += m_PhysDataNum;
  m_MaterialData = valueptr;
  ZeroPhysDatas();
}



void DataBase::CleanPhysDataPtr()
{
  m_PhysDataNum = 0.0;
  m_PhysData = NULL;
}


