#include <DynFlds_Voltage_Dgn.hxx>
#include <stdlib.h>


DynFlds_Voltage_Dgn::
DynFlds_Voltage_Dgn()
  :FieldsDgnBase()
{
  m_Voltage = 0.0;
}


void
DynFlds_Voltage_Dgn::
Init(const FieldsDefineCntr* theCntr)
{
  FieldsDgnBase::Init(theCntr);
  m_Voltage = 0.0;
}


DynFlds_Voltage_Dgn::
~DynFlds_Voltage_Dgn()
{
}


void 
DynFlds_Voltage_Dgn::
SetAttrib(const TxHierAttribSet& tha)
{
  if(tha.hasOption("lineDir")){
    m_LineDir = tha.getOption("lineDir");
  }else{
    cout<<"DynFlds_Voltage_Dgn::SetAttrib--------------error----lineDir"<<endl;
  }


  TxVector2D<Standard_Real> theOrg;
  if(tha.hasPrmVec("org")){
    vector<Standard_Real> theData = tha.getPrmVec("org");
    if(theData.size()>=2){
      theOrg[0] = theData[0];
      theOrg[1] = theData[1];
    }else{
      cout<<"DynFlds_Voltage_Dgn::SetAttrib--------------error----org"<<endl;
    }
  }
  
  Standard_Size PhiNumber = GetGridGeom_Cyl3D()->GetDimPhi();
  if(PhiNumber == 1)
  {
  	m_PhiIndex = -1;
  }
  else if(tha.hasOption("Phi")){
    m_PhiIndex= tha.getOption("Phi");
  }else{
    cout<<"DynFlds_VoltageData_Dgn::SetAttrib--------------error----Phi"<<endl;
  }


  if(GetFldsDefCntr()->GetZRGrid()->IsIn(theOrg)){ 
    GetFldsDefCntr()->GetZRGrid()->ComputeLocationInGrid(theOrg, m_OrgIndx);
  }else{
    cout<<"DynFlds_Voltage_Dgn::SetAttrib--------------error----org------is not in simulation rgn"<<endl;
    exit(1);
  }

  TxVector2D<Standard_Real> theEnd;
  if(tha.hasPrmVec("end")){
    vector<Standard_Real> theData = tha.getPrmVec("end");
    if(theData.size()>=2){
      theEnd[0] = theData[0];
      theEnd[1] = theData[1];
    }else{
      cout<<"DynFlds_Voltage_Dgn::SetAttrib--------------error----end"<<endl;
    }
  }

  if(GetFldsDefCntr()->GetZRGrid()->IsIn(theEnd)){ 
    GetFldsDefCntr()->GetZRGrid()->ComputeLocationInGrid(theEnd, m_EndIndx);
  }else{
    cout<<"DynFlds_Voltage_Dgn::SetAttrib--------------error----end------is not in simulation rgn"<<endl;
    exit(1);
  }

  /*
  theOrg.write(cout);
  m_OrgIndx.write(cout);

  theEnd.write(cout);
  m_EndIndx.write(cout);
  //*/

  string theName="";
  if(tha.hasString("name")){
    theName = tha.getString("name");
  }else{
    cout<<"DynFlds_Voltage_Dgn::SetAttrib--------------error----name"<<endl;
  }
  SetName(theName);  // dynobj

  InitData();
}



void 
DynFlds_Voltage_Dgn::
InitData()
{
  Standard_Integer Dir1=(m_LineDir+1)%2;
  Standard_Integer theLineOrgIndx = m_OrgIndx[m_LineDir]-1;
  Standard_Integer theLineEndIndx = m_EndIndx[m_LineDir]+1;

  TxSlab2D<Standard_Integer> thePhysRgn= GetFldsDefCntr()->GetZRGrid()->GetPhysRgn();

  TxSlab2D<Standard_Integer> tmpRgn;
  tmpRgn.setLowerBound(Dir1, m_OrgIndx[Dir1]);
  tmpRgn.setUpperBound(Dir1, m_OrgIndx[Dir1]);
    
    
  tmpRgn.setLowerBound(m_LineDir,theLineOrgIndx);
  tmpRgn.setUpperBound(m_LineDir,theLineEndIndx);
    

  TxSlab2D<Standard_Integer> theRgn = thePhysRgn & tmpRgn;

  GetFldsDefCntr()->GetGridGeom(m_PhiIndex)->GetGridEdgeDatasNotOfMaterialTypeOfSubRgn( (Standard_Integer)PEC, theRgn,false, m_Datas);

}



void 
DynFlds_Voltage_Dgn::
ComputeVoltage()
{
  m_Voltage = 0.0;

  Standard_Integer thePhysDataIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
  for(Standard_Size n=0; n<m_Datas.size(); n++){
    m_Voltage += m_Datas[n]->GetPhysData(thePhysDataIndex) * m_Datas[n]->GetLength();
  }

}






void 
DynFlds_Voltage_Dgn::
Advance()
{
  ComputeVoltage();
  DynObj::Advance();
}



Standard_Real 
DynFlds_Voltage_Dgn::
GetValue()
{
  return m_Voltage;
}
