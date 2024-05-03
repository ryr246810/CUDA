#include <TxStreams.h>
#include <TxHierAttribSet.h>

#include <TxMaker.h>
#include <TxMakerMap.h>
#include <TxMakerMapBase.h>



#include <ComboFields_Dynamic_VertexElecSrc.hxx>


void 
ComboFields_Dynamic_VertexElecSrc::
SetAttrib(const TxHierAttribSet& tha)
{
  for(Standard_Size i=0; i<2; i++){
    m_VertexIndex[i] = 0;
  }

  if(tha.hasPrmVec("location")){
    Standard_Real theLocation[2];
    vector<Standard_Real> theData = tha.getPrmVec("location");
    if(theData.size()>=2){
      theLocation[0] = theData[0];
      theLocation[1] = theData[1];
    }else{
      cout<<"ComboFields_Dynamic_VertexElecSrc::SetAttrib--------------error----location"<<endl;
    }
    GetFldsDefCntr()->GetZRGrid()->ComputeLocationInGrid(theLocation, m_VertexIndex);
  }else if(tha.hasOptVec("locationIndex")){
    vector<int> theindex = tha.getOptVec("locationIndex");
    if(theindex.size()>=2){
      m_VertexIndex[0] = theindex[0];
      m_VertexIndex[1] = theindex[1];
    }else{
      for(Standard_Size i=0; i<theindex.size(); i++){
	m_VertexIndex[i] = theindex[i];
      }
    }
  }else{
    cout<<"ComboFields_Dynamic_VertexElecSrc::SetAttrib--------------error----locationIndex-----2"<<endl;
  }


  //cout<<"m_VertexIndex = ["<<m_VertexIndex[0]<<", "<<m_VertexIndex[1]<<"]"<<endl;
  Standard_Size PhiNumber = GetGridGeom_Cyl3D()->GetDimPhi();
  if(PhiNumber == 1)
  {
  	m_phiIndex = -1;
  }
  else if(tha.hasOption("phiIndex")){
    m_phiIndex = tha.getOption("phiIndex");
    if(m_phiIndex>=PhiNumber)
    {
    	cout<<"error inComboFields_Dynamic_VertexElecSrc::SetAttrib()"<<endl;
    	cout<<"phiIndex >= PhiNumber"<<endl;
    	exit(1);
    }
  }else{
    m_phiIndex = 0;
  }

  if(tha.hasParam("startTime")){
    m_StartTime = tha.getParam("startTime");
  }else{
    m_StartTime = 0.;
  }

  if(tha.hasParam("endTime")){
    m_EndTime = tha.getParam("endTime");
  }else{
    m_EndTime = 0.;
  }

  SetupRgn();

  /*************************************Port Func ***********************************************/
  std::vector< std::string > tfuncNames = tha.getNamesOfType("TFunc");

  // Go through the list and test the functors
  if(!tfuncNames.size()){
    m_tfuncPtr = new TFunc;;
    std::cout << "No temporal Function specified.\n";
  }else{
    // Get attributes of functor and name of function
    TxHierAttribSet attribs = tha.getAttrib(tfuncNames[0]);
    string functionName = attribs.getString("function");

    // Create this functor
    try {
      m_tfuncPtr = TxMakerMap<TFunc>::getNew(functionName);
    }
    catch (TxDebugExcept& txde) {
      std::cout << txde << std::endl;
      return;
    }

    // Set attributes of functor
    m_tfuncPtr->setAttrib(attribs);
  }

}

