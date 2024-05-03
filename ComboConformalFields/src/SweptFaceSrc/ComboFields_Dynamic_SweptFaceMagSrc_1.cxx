#include <TxStreams.h>
#include <TxHierAttribSet.h>

#include <TxMaker.h>
#include <TxMakerMap.h>
#include <TxMakerMapBase.h>



#include <ComboFields_Dynamic_SweptFaceMagSrc.hxx>


void 
ComboFields_Dynamic_SweptFaceMagSrc::
SetAttrib(const TxHierAttribSet& tha)
{
  if(tha.hasOption("dir")){
    m_Dir = tha.getOption("dir");
  }

  for(Standard_Size i=0; i<2; i++){
    m_FirstVertexIndex[i] = 0;
  }


  if(tha.hasPrmVec("location")){
    Standard_Real theLocation[2];
    vector<Standard_Real> theData = tha.getPrmVec("location");
    if(theData.size()>=2){
      theLocation[0] = theData[0];
      theLocation[1] = theData[1];
    }else{
      cout<<"ComboFields_Dynamic_SweptFaceMagSrc::SetAttrib--------------error----location"<<endl;
    }
    GetFldsDefCntr()->GetZRGrid()->ComputeLocationInGrid(theLocation, m_FirstVertexIndex);
  }else if(tha.hasOptVec("locationIndex")){
    vector<int> theindex = tha.getOptVec("locationIndex");
    if(theindex.size()>=2){
      m_FirstVertexIndex[0] = theindex[0];
      m_FirstVertexIndex[1] = theindex[1];
    }else{
      for(Standard_Size i=0; i<theindex.size(); i++){
	      m_FirstVertexIndex[i] = theindex[i];
      }
    }
  }else{
    cout<<"ComboFields_Dynamic_SweptFaceMagSrc::SetAttrib--------------error----locationIndex-----2"<<endl;
  }


  if(tha.hasOption("occupyNum")){
    m_GridEdgeNum = tha.getOption("occupyNum");
  }else{
    m_GridEdgeNum = 1;
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

