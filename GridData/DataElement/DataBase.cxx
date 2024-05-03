#include <DataBase.hxx>
#include <GeomDataBase.hxx>

DataBase::DataBase()
  :GeomDataBase()
{
  m_MaterialType=0;
  m_PhysDataNum=0;
  m_PhysData=NULL;

  m_SweptPhysDataNum = 0;
  m_SweptPhysData=NULL;

  m_MaterialData=NULL;
  m_MaterialDataInv=NULL;
  m_PMLData=NULL;
}



DataBase::DataBase( Standard_Integer _mark)
  :GeomDataBase(_mark)
{
  m_MaterialType=0;
  m_PhysDataNum=0;
  m_PhysData=NULL;

  m_SweptPhysDataNum = 0;
  m_SweptPhysData=NULL;

  m_MaterialData=NULL;
  m_MaterialDataInv=NULL;
  m_PMLData=NULL;
}



DataBase::~DataBase()
{
  ClearPhysData();
  ClearSweptPhysData();
  ClearMaterialData();
  ClearPMLData();
}



void DataBase::Setup()
{
  SetupGeomDimInf();
  SetupMaterialData();
}


/***************************************************************/
/***************************************************************/
/***************************************************************/
/***************************************************************/
/***************************************************************/
void DataBase::ClearMaterialData()
{
  if(m_MaterialData!=NULL){
    delete[] m_MaterialData;
    delete[] m_MaterialDataInv;
    m_MaterialData=NULL;
    m_MaterialDataInv = NULL;
  }
}


void DataBase::SetupMaterialData()
{
  if(m_MaterialData!=NULL){
    ClearMaterialData();
  }else{
    m_MaterialData = new Standard_Real[3];
    m_MaterialDataInv = new Standard_Real[3];
    for(Standard_Integer i=0; i<3; i++){
      m_MaterialData[i] = 0.0;
      m_MaterialDataInv[i] = 0.0;
    }
  }
}


bool DataBase::IsMaterialDataDefined() const
{
  bool result = true;
  if(m_MaterialData==NULL){
    result = false;
  }
  return result;
}



/***************************************************************/
/***************************************************************/
/***************************************************************/
/***************************************************************/
/***************************************************************/
bool 
DataBase::
IsMaterialType(const Standard_Integer _materialtype) const
{
  bool result = false;
  if( (m_MaterialType & _materialtype) != 0) {
    result = true;
  }
  return result;
}

void DataBase::AddMaterialType(const Standard_Integer _materialtype)
{
  m_MaterialType =  m_MaterialType | _materialtype;
}

void DataBase::DelMaterialType(const Standard_Integer _materialtype)
{
  Standard_Integer tmpMaterialType = m_MaterialType & _materialtype;
  m_MaterialType -= tmpMaterialType;
}

void DataBase::SetMaterialType(const Standard_Integer _materialtype)
{
  m_MaterialType = _materialtype;
}

Standard_Integer DataBase::GetMaterialType() const
{
  return m_MaterialType;
}

void DataBase::ResetMaterialType()
{
  SetMaterialType(0);
}

void DataBase::ResetEMMaterialType()
{
  m_MaterialType &= EM_ZERO_MASK;
}

void DataBase::ResetPtclMaterialType()
{
  m_MaterialType&= PTCL_ZERO_MASK;
}

Standard_Integer DataBase::GetEMMaterialType()  const
{
  Standard_Integer em_mark = m_MaterialType & EM_ONLY_MASK;
  return em_mark;
}

Standard_Integer DataBase::GetPtclMaterialType() const
{
  Standard_Integer ptcl_mark =  m_MaterialType & PTCL_ONLY_MASK;
  return ptcl_mark;
}
/***************************************************************/
/***************************************************************/
/***************************************************************/
/***************************************************************/
/***************************************************************/

bool DataBase::BeInPMLRegion() const
{
  bool result = false;
  if( (GetEMMaterialType() & PML)!=0) result =  true;
  return result;
}


bool DataBase::IsPMLDataDefined() const
{
  bool result = true;
  if(m_PMLData==NULL){
    result = false;
  }
  return result;
}


void DataBase::LocateMemeoryForPMLData()
{
  if( BeInPMLRegion() ){
    m_PMLData = new Standard_Real[10];

    // sigma
    for(Standard_Integer i=0;i<2;i++){
      m_PMLData[i]=0;
    }

    // alpha
    for(Standard_Integer i=0;i<2;i++){
      Standard_Integer j = i+2;
      m_PMLData[j]=0;
    }

    // kappa
    for(Standard_Integer i=0;i<2;i++){
      Standard_Integer k = i+4;
      m_PMLData[k]=1;
    }

    // a
    for(Standard_Integer i=0;i<2;i++){
      Standard_Integer l = i+6;
      m_PMLData[l]=0;
    }

    // b
    for(Standard_Integer i=0;i<2;i++){
      Standard_Integer m = i+8;
      m_PMLData[m]=0;
    }
  }
}


void DataBase::SetupPMLData()
{
  if( !IsPMLDataDefined() ){
    LocateMemeoryForPMLData();
  }
}


void DataBase::ResetPMLData()
{
  ClearPMLData();
}


void DataBase::ClearPMLData()
{
  if(m_PMLData!=NULL){
    delete[] m_PMLData;
    m_PMLData=NULL;
  }
}


Standard_Real DataBase::GetPMLSigma(Standard_Integer _dir)
{
  Standard_Real result = 0.0;
  if( BeInPMLRegion() ){
    result = m_PMLData[_dir];
  }
  return result;
}


void DataBase::SetPMLSigma(Standard_Integer _dir, Standard_Real _pmlsigma)
{
  if( BeInPMLRegion() ){
    m_PMLData[_dir] =  _pmlsigma;
  }
}


Standard_Real DataBase::GetPMLAlpha(Standard_Integer _dir)
{
  Standard_Real result = 0.0;
  Standard_Integer theAlphaIndex = _dir+2;
  if( BeInPMLRegion() ){
    result = m_PMLData[theAlphaIndex];
  }
  return result;
}


void DataBase::SetPMLAlpha(Standard_Integer _dir, Standard_Real _pmlalpha)
{
  Standard_Integer theAlphaIndex = _dir+2;
  if( BeInPMLRegion() ){
    m_PMLData[theAlphaIndex] =  _pmlalpha;
  }
}


Standard_Real DataBase::GetPMLKappa(Standard_Integer _dir)
{
  Standard_Real result = 1.;
  Standard_Integer theKappaIndex = _dir+4;
  if( BeInPMLRegion() ){
    result = m_PMLData[theKappaIndex];
  }
  return result;
}


void DataBase::SetPMLKappa(Standard_Integer _dir, Standard_Real _pmlkappa)
{
  Standard_Integer theKappaIndex = _dir+4;
  if( BeInPMLRegion() ){
    m_PMLData[theKappaIndex] =  _pmlkappa;
  }
}


Standard_Real DataBase::GetPML_a(Standard_Integer _dir)
{
  Standard_Real result = 0.0;
  Standard_Integer theIndex = _dir+6;
  if( BeInPMLRegion() ){
    result = m_PMLData[theIndex];
  }
  return result;
}


void DataBase::SetPML_a(Standard_Integer _dir, Standard_Real _a)
{
  Standard_Integer theIndex = _dir+6;
  if( BeInPMLRegion() ){
    m_PMLData[theIndex] =  _a;
  }
}


Standard_Real DataBase::GetPML_b(Standard_Integer _dir)
{
  Standard_Real result = 0.0;
  Standard_Integer theIndex = _dir+8;
  if( BeInPMLRegion() ){
    result = m_PMLData[theIndex];
  }
  return result;
}


void DataBase::SetPML_b(Standard_Integer _dir, Standard_Real _b)
{
  Standard_Integer theIndex = _dir+8;
  if( BeInPMLRegion() ){
    m_PMLData[theIndex] = _b;
  }
}


/***************************************************************/
/***************************************************************/
/***************************************************************/
/***************************************************************/
/***************************************************************/


void 
DataBase::
SetEpsilon(const Standard_Real _epsilon)
{
  m_MaterialData[0] = _epsilon;
  m_MaterialDataInv[0] = 1 / _epsilon;
}


void 
DataBase::
SetMu(const Standard_Real _mu)
{
  m_MaterialData[1] = _mu;
  m_MaterialDataInv[1] = 1 / _mu;
}


void 
DataBase::
SetSigma(const Standard_Real _sigma)
{
  m_MaterialData[2] = _sigma;
  m_MaterialDataInv[2] = 1 / _sigma;
}


Standard_Real 
DataBase::
GetEpsilon() const
{
  return m_MaterialData[0];
}


Standard_Real 
DataBase::
GetMu() const
{
  return m_MaterialData[1];
}


Standard_Real 
DataBase::
GetSigma() const
{
  return m_MaterialData[2];
}

Standard_Real 
DataBase::
GetEpsilonInv() const
{
  return m_MaterialDataInv[0];
}


Standard_Real 
DataBase::
GetMuInv() const
{
  return m_MaterialDataInv[1];
}


Standard_Real 
DataBase::
GetSigmaInv() const
{
  return m_MaterialDataInv[2];
}
