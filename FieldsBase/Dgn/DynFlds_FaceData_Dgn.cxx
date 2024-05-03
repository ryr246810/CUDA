#include <DynFlds_FaceData_Dgn.hxx>
#include <PhysConsts.hxx>



DynFlds_FaceData_Dgn::
DynFlds_FaceData_Dgn()
  :FieldsDgnBase()
{
  m_BaseFace = NULL;
  m_Data = 0.0;
}



void
DynFlds_FaceData_Dgn::
Init(const FieldsDefineCntr* theCntr)
{
  FieldsDgnBase::Init(theCntr);
  m_Data = 0.0;
}



DynFlds_FaceData_Dgn::~DynFlds_FaceData_Dgn()
{
}



void 
DynFlds_FaceData_Dgn::
SetAttrib(const TxHierAttribSet& tha)
{
  Standard_Integer theCutIndex = 0;
  string theName="";

  Standard_Size PhiNumber = GetGridGeom_Cyl3D()->GetDimPhi();
  if(PhiNumber == 1)
  {
  	m_PhiIndex = -1;
  }
  else if(tha.hasOption("Phi")){
    m_PhiIndex= tha.getOption("Phi");
  }else{
    cout<<"DynFlds_FaceData_Dgn::SetAttrib--------------error----Phi"<<endl;
  }

  Standard_Size faceIndxVec[2];
  if(tha.hasPrmVec("location")){
    Standard_Real theLocation[2];
    vector<Standard_Real> theData = tha.getPrmVec("location");
    if(theData.size()>=2){
      theLocation[0] = theData[0];
      theLocation[1] = theData[1];
    }else{
      cout<<"DynFlds_FaceData_Dgn::SetAttrib--------------error----location"<<endl;
    }
    GetFldsDefCntr()->GetZRGrid()->ComputeLocationInGrid(theLocation, faceIndxVec);
  }else if(tha.hasOptVec("locationIndex")){
    vector<int> theindex = tha.getOptVec("locationIndex");
    if(theindex.size()>=2){
      faceIndxVec[0] = theindex[0];
      faceIndxVec[1] = theindex[1];
    }else{
      cout<<"DynFlds_FaceData_Dgn::SetAttrib--------------error----locationIndex-----1"<<endl;
    }
  }else{
    cout<<"DynFlds_FaceData_Dgn::SetAttrib--------------error----locationIndex-----2"<<endl;
  }

  if(tha.hasString("name")){
    theName = tha.getString("name");
  }else{
    cout<<"DynFlds_FaceData_Dgn::SetAttrib--------------error----name"<<endl;
  }

  SetName(theName);  // dynobj

  InitParamt(faceIndxVec);  // poynting
}



void 
DynFlds_FaceData_Dgn::
InitParamt(const Standard_Size faceIndxVec[2])
{

  TxSlab2D<Standard_Integer> theRgn  = GetFldsDefCntr()->GetZRGrid()->GetPhysRgn();

  if( ( (faceIndxVec[0]>=theRgn.getLowerBound(0)) && (faceIndxVec[0]<theRgn.getUpperBound(0)) ) &&
      ( (faceIndxVec[1]>=theRgn.getLowerBound(1)) && (faceIndxVec[1]<theRgn.getUpperBound(1)) ) ) { 

    Standard_Size currFaceIndex;
    GetFldsDefCntr()->GetZRGrid()->FillFaceIndx(faceIndxVec, currFaceIndex);
    
    m_BaseFace = GetGridGeom(m_PhiIndex)->GetGridFaces() + currFaceIndex;
  }else{
    m_BaseFace = NULL;
  }
}



Standard_Real 
DynFlds_FaceData_Dgn::
GetValue()
{
  return m_Data*mksConsts.mu0;
}


void 
DynFlds_FaceData_Dgn::
ComputeData()
{
  m_Data = 0.0;

  Standard_Integer facePhysDataIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicMagField_PhysDataIndex();

  if(m_BaseFace!=NULL){
    const vector<GridFaceData*>& currFaces = m_BaseFace->GetFaces();
    size_t nb = currFaces.size();
    for(size_t i=0; i<nb; i++){
      m_Data += currFaces[i]->GetPhysData(facePhysDataIndex)/nb;
    }
  }
}


void
DynFlds_FaceData_Dgn::
Advance()
{
  ComputeData();
  DynObj::Advance();
}
