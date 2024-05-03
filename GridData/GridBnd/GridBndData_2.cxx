
#include <GridBndData.hxx>
#include <BaseDataDefine.hxx>
#include <PortDataFunc.hxx>
#include <stdlib.h>
#include <set>


void
GridBndData::
SetAttrib(const std::string& workPath,  const TxHierAttribSet& tas)
{
  if(m_BackGroundMaterialType==USERDEFINED){
    if(tas.hasOption("backGroundMaterialDataIndex")){
      m_BackGroundMaterialDataIndex = tas.getOption("backGroundMaterialDataIndex");
    }else{
      cout<<"error--------------------GridBndData::SetAttrib-----------background data nodefined----2"<<endl;
    }
  }else{
    m_BackGroundMaterialDataIndex = -1;  // no used material data index, only for valcumm;
  }


  {
    // for all material data defination
    std::vector< std::string > theMatDataNames = tas.getNamesOfType("MaterialDatas");
    if(theMatDataNames.size()>0){
      TxHierAttribSet tha = tas.getAttrib(theMatDataNames[0]);
      SetMaterialDatas(tha);
    }
  }


  {
    // build the map of the shape index and the material data index
    std::vector< std::string > theShapeMatDataNames = tas.getNamesOfType("ShapeMaterialMap");
    if(theShapeMatDataNames.size()>0){
      TxHierAttribSet tha = tas.getAttrib(theShapeMatDataNames[0]);
      SetShapeMaterialIndexMap(tha);
    }
  }


  { // add 2016.08.26
    // for all material data defination
    std::vector< std::string > theMurPortDataNames = tas.getNamesOfType("MurPortDatas");
    if(theMurPortDataNames.size()>0){
      TxHierAttribSet tha = tas.getAttrib(theMurPortDataNames[0]);
      SetMurPortDatas(tha);
    }
  }


  // the material data indices only for space defination (userdefined backgound)
  BuildMatDataIndicesOfSpaceDefine();

  bool checkResult = CheckMatDataIndices();
  if(!checkResult){
    cout<<"error-----------------GridBndData::SetAttrib-------------------material data indices are not defined correctly, program will exit"<<endl;
    exit(1);
  }
}


void 
GridBndData::
SetShapeMaterialIndexMap(const TxHierAttribSet& tas)
{
  m_ShapeWithMaterialDataIndexMap.clear();

  std::vector< std::string > theDataNames = tas.getNamesOfType("ShapeMaterialDefine");

  /*
  if( theDataNames.size() ){
    std::cout << "ShapeMaterialDefines are:";
    for(size_t i=0; i<theDataNames.size(); ++i)
      std::cout << " " << theDataNames[i];
    std::cout << std::endl;
  }
  //*/

  // Add in all the ports
  for(size_t i=0; i<theDataNames.size(); ++i){
    TxHierAttribSet tha = tas.getAttrib(theDataNames[i]);

    Standard_Integer theShapeIndex = 0;

    if(tha.hasOption("mask")){
      Standard_Integer tmpMask = tha.getOption("mask");
      this->ConvertShapeMasktoIndex(tmpMask, theShapeIndex);
    }else{
      cout<<"error--------------------GridBndData::SetMaterialDatas------------mask not be defined"<<endl;
      continue;
    }

    Standard_Integer theMaterialDataIndex = -1;
    if(tha.hasOption("materialDataIndex")){
      theMaterialDataIndex = tha.getOption("materialDataIndex");
    }else{
      cout<<"error--------------------GridBndData::SetShapeMaterialIndexMap-----------user defined material data nodefined"<<endl;
    }

    m_ShapeWithMaterialDataIndexMap.insert( pair<Standard_Integer, Standard_Integer>(theShapeIndex, theMaterialDataIndex) );
  }
}


void 
GridBndData::
SetMaterialDatas(const TxHierAttribSet& tas)
{
  m_MaterialDataIndexWithMaterialDataMap.clear();

  std::vector< std::string > theDataNames = tas.getNamesOfType("OneMaterialData");

  /*
  if( theDataNames.size() ){
    std::cout << "MaterialData are:";
    for(size_t i=0; i<theDataNames.size(); ++i)
      std::cout << " " << theDataNames[i];
    std::cout << std::endl;
  }
  //*/

  // Add in all the ports
  for(size_t i=0; i<theDataNames.size(); ++i){
    TxHierAttribSet tha = tas.getAttrib(theDataNames[i]);

    Standard_Integer theMaterialDataIndex = -2;
    if(tha.hasOption("materialDataIndex")){
      theMaterialDataIndex = tha.getOption("materialDataIndex");
    }else{
      cout<<"error--------------------GridBndData::SetMaterialDatas-----------user defined material data nodefined"<<endl;
    }

    ISOEMMatData tmpData;

    if(tha.hasParam("eps_z")){
      tmpData.m_eps_z = tha.getParam("eps_z");
    }else{
      tmpData.m_eps_z = 1.0;
    }
    
    if(tha.hasParam("eps_r")){
      tmpData.m_eps_r = tha.getParam("eps_r");
    }else{
      tmpData.m_eps_r = 1.0;
    }

    if(tha.hasParam("eps_Phi")){
      tmpData.m_eps_Phi = tha.getParam("eps_Phi");
    }else{
      tmpData.m_eps_Phi = 1.0;
    }

    if(tha.hasParam("mu_z")){
      tmpData.m_mu_z = tha.getParam("mu_z");
    }else{
      tmpData.m_mu_z = 1.0;
    }

    if(tha.hasParam("mu_r")){
      tmpData.m_mu_r = tha.getParam("mu_r");
    }else{
      tmpData.m_mu_r = 1.0;
    }
    if(tha.hasParam("mu_Phi")){
      tmpData.m_mu_Phi = tha.getParam("mu_Phi");
    }else{
      tmpData.m_mu_Phi = 1.0;
    }

    if(tha.hasParam("sigma_z")){
      tmpData.m_sigma_z = tha.getParam("sigma_z");
    }else{
      tmpData.m_sigma_z = 0.0;
    }

    if(tha.hasParam("sigma_r")){
      tmpData.m_sigma_r = tha.getParam("sigma_r");
    }else{
      tmpData.m_sigma_r = 0.0;
    }

    if(tha.hasParam("sigma_Phi")){
      tmpData.m_sigma_Phi = tha.getParam("sigma_Phi");
    }else{
      tmpData.m_sigma_Phi = 0.0;
    }

    if( (m_MaterialDataIndexWithMaterialDataMap.find(theMaterialDataIndex)) == (m_MaterialDataIndexWithMaterialDataMap.end()) ){
      m_MaterialDataIndexWithMaterialDataMap.insert( pair<Standard_Integer, ISOEMMatData>(theMaterialDataIndex, tmpData) );
    }else{
      cout<<"GridBndData::SetMaterialDatas----------duplate defination of material data----------error"<<endl;
    }

  }
}



void 
GridBndData::
SetMurPortDatas(const TxHierAttribSet& tas)
{
  m_MurPortIndexWithDataMap.clear();

  std::vector< std::string > theDataNames = tas.getNamesOfType("OneMurPortData");

  /*
  if( theDataNames.size() ){
    std::cout << "MurPortData are:";
    for(size_t i=0; i<theDataNames.size(); ++i)
      std::cout << " " << theDataNames[i];
    std::cout << std::endl;
  }
  //*/

  // Add in all the ports
  for(size_t i=0; i<theDataNames.size(); ++i){
    TxHierAttribSet tha = tas.getAttrib(theDataNames[i]);

    Standard_Integer thePortMask = -2;
    Standard_Integer thePortIndex = -2;
    if(tha.hasOption("mask")){
      thePortMask = tha.getOption("mask");
      ConvertFaceMasktoIndex(thePortMask, thePortIndex);
    }else{
      cout<<"error--------------------GridBndData::SetMurPortDatas-----------user defined material data nodefined"<<endl;
    }

    Standard_Real theVp;
    if(tha.hasParam("vp")){
      theVp = tha.getParam("vp");
    }else{
      theVp = 1.0;
    }

    const PortData* thePort = GetPortWithPortIndex(thePortIndex);
    if(IsMurPortType(thePort->m_Type)){
      if( (m_MurPortIndexWithDataMap.find(thePortIndex)) == (m_MurPortIndexWithDataMap.end()) ){
	m_MurPortIndexWithDataMap.insert( pair<Standard_Integer, Standard_Real>(thePortIndex, theVp) );
      }else{
	cout<<"GridBndData::SetMurPortDatas----------duplate defination of mur port data----------error"<<endl;
      }
    }
  }
}




void
GridBndData::
BuildMatDataIndicesOfSpaceDefine()
{
  m_SpaceMaterialData.clear();


  if(m_BackGroundMaterialType==USERDEFINED){
    m_SpaceMaterialData.push_back(m_BackGroundMaterialDataIndex);
  }
}



bool
GridBndData::
CheckMatDataIndices() const
{
  bool result = true;

  set<Standard_Integer> theAllMatDataIndices;
  theAllMatDataIndices.clear();

  map<Standard_Integer, ISOEMMatData, less<Standard_Integer> >::const_iterator iter;
  for(iter = m_MaterialDataIndexWithMaterialDataMap.begin(); iter!=m_MaterialDataIndexWithMaterialDataMap.end(); iter++){
    Standard_Integer currIndex = iter->first;
    theAllMatDataIndices.insert(currIndex);
  }


  // check background material data defination
  if(m_BackGroundMaterialType==USERDEFINED){
    if(theAllMatDataIndices.count(m_BackGroundMaterialDataIndex)==0){
      result = false;
    }
  }


  // check shape's material data defination
  if(result){
    map<Standard_Integer, Standard_Integer, less<Standard_Integer> >::const_iterator iter0;
    for(iter0 = m_ShapeWithMaterialDataIndexMap.begin(); iter0!=m_ShapeWithMaterialDataIndexMap.end(); iter0++){
      Standard_Integer currIndex = iter0->second;
      if(theAllMatDataIndices.count(currIndex)==0){
	result = false;
	break;
      }
    } 
  }


  return result;
}
