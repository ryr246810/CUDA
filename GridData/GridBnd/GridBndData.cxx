#include <GridBndData.hxx>

GridBndData::
GridBndData()
{
}


GridBndData::
~GridBndData()
{
  m_EdgeBndVertexData0.clear();
  m_EdgeBndVertexData1.clear();

  m_FaceBndVertexData.clear();


  m_ShapesWithTypeTool.clear();
  m_FacesWithTypeTool.clear();
  m_FacesWithShapeTool.clear();


  m_ShapeMaskWithIndexTool.clear();
  m_FaceMaskWithIndexTool.clear();


  m_Ports.clear();


  m_MaterialDataIndexWithMaterialDataMap.clear(); // 2015.09.29
  m_ShapeWithMaterialDataIndexMap.clear(); // 2015.09.29


  m_MurPortIndexWithDataMap.clear(); //2016.08.26
}


const map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > * 
GridBndData::
GetEdgeBndVertexDataOf(const ZRGridLineDir aDir) const
{
  const map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > * theVData;
  switch (aDir)
    {
    case DIRRZZ:
      {
	theVData = &m_EdgeBndVertexData0;
	break;
      }
    case DIRRZR:
      {
	theVData = &m_EdgeBndVertexData1;
	break;
      }
    }
  return theVData;
}


map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > * 
GridBndData::ModifyEdgeBndVertexDataOf(const ZRGridLineDir aDir)
{
  map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > * theVData;
  switch (aDir)
    {
    case DIRRZZ:
      {
	theVData = &m_EdgeBndVertexData0;
	break;
      }
    case DIRRZR:
      {
	theVData = &m_EdgeBndVertexData1;
	break;
      }
    }
  return theVData;
}


const vector<FaceBndVertexData> *    
GridBndData::GetFaceBndVertexData() const 
{
  const vector<FaceBndVertexData> * theCVData = &m_FaceBndVertexData;
  return theCVData;
}


vector<FaceBndVertexData> *    
GridBndData::ModifyFaceBndVertexData()
{
  vector<FaceBndVertexData> * theCVData = &m_FaceBndVertexData;
  return theCVData;
}


const vector<EdgeBndVertexData>& 
GridBndData::
GetEdgeBndVertexDataOf(const ZRGridLineDir aDir, 
		       const Standard_Size anIndex) const
{
  const map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > * theVData = GetEdgeBndVertexDataOf(aDir);
  map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > ::const_iterator iter = theVData->find(anIndex);
  return iter->second;
}


vector<EdgeBndVertexData>& 
GridBndData::
ModifyEdgeBndVertexDataOf(const ZRGridLineDir aDir,
			  const Standard_Size anIndex)
{
  map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > * theVData = ModifyEdgeBndVertexDataOf(aDir);
  map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > ::iterator iter = theVData->find(anIndex);
  return iter->second;
}


bool 
GridBndData::
HasEdgeBndVertexDataOf(const ZRGridLineDir aDir, 
		       const Standard_Size anIndex) const
{
  bool result = false;
  const map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > * theVData = GetEdgeBndVertexDataOf(aDir);
  if(theVData->size()>0){
    map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> > ::const_iterator iter = theVData->find(anIndex);
    if(iter!=theVData->end()) result=true;
  }
  return result;
}


const map<Standard_Integer, PortData, less<Standard_Integer> >* 
GridBndData::
GetPorts() const
{
  return &m_Ports;
}


map<Standard_Integer, PortData, less<Standard_Integer> >* 
GridBndData::
ModifyPorts()
{
  return &m_Ports;
}


const map<Standard_Integer, Standard_Real, less<Standard_Integer> >* 
GridBndData::
GetMurPortDatas() const
{
  return &m_MurPortIndexWithDataMap;
}


Standard_Integer 
GridBndData::
GetMaterialTypeWithShapeIndex(const Standard_Integer theIndex) const
{
  Standard_Integer result = 0;
  map<Standard_Integer, Standard_Integer>::const_iterator iter = m_ShapesWithTypeTool.find(theIndex);
  if(iter != m_ShapesWithTypeTool.end()){
    result = iter->second;
  }
  return result;
}



Standard_Integer 
GridBndData::
GetMaterialTypeWithShapeIndices(const set<Standard_Integer>& theIndices) const
{
  Standard_Integer result = 0;

  set<Standard_Integer>::const_iterator iter;
  for(iter=theIndices.begin();iter!=theIndices.end(); iter++){
    Standard_Integer theIndex = *iter;
    map<Standard_Integer, Standard_Integer>::const_iterator iter = m_ShapesWithTypeTool.find(theIndex);
    if(iter != m_ShapesWithTypeTool.end()){
      result = result|iter->second;
    }
  }

  return result;
}



Standard_Integer 
GridBndData::
GetMaterialTypeWithFaceIndices(const set<Standard_Integer>& theIndices) const
{
  Standard_Integer result = 0;

  set<Standard_Integer>::const_iterator iter;
  for(iter=theIndices.begin();iter!=theIndices.end(); iter++){
    Standard_Integer theIndex = *iter;
    map<Standard_Integer, Standard_Integer>::const_iterator iter = m_FacesWithTypeTool.find(theIndex);
    if(iter != m_FacesWithTypeTool.end()){
      result = result|iter->second;
    }
  }

  return result;
}



Standard_Integer 
GridBndData::
GetMaterialTypeWithFaceIndex(const Standard_Integer theIndex) const
{
  Standard_Integer result = 0;
    map<Standard_Integer, Standard_Integer>::const_iterator iter =  m_FacesWithTypeTool.find(theIndex);

  if(iter != m_FacesWithTypeTool.end()){
    result = iter->second;
  }
  return result;
}


const PortData* 
GridBndData::
GetPortWithPortIndex(const Standard_Integer thePortIndex) const
{
  const PortData* result = NULL;
  const map<Standard_Integer, PortData, less<Standard_Integer> >* thePorts = GetPorts();

  map<Standard_Integer, PortData, less<Standard_Integer> >::const_iterator iter = thePorts->find(thePortIndex);
  if(iter!=thePorts->end()){
    result = &(iter->second);
  }
  return result;
}



Standard_Integer 
GridBndData::
GetShapeIndexAccordingFaceIndex(const Standard_Integer theIndex) const
{
  Standard_Integer result = 0;
  map<Standard_Integer, Standard_Integer>::const_iterator iter = m_FacesWithShapeTool.find(theIndex);
  if(iter!=m_FacesWithShapeTool.end()){
    result = iter->second;
  }
  return result;
}


const map<Standard_Integer, Standard_Integer>* 
GridBndData::
GetShapesType() const 
{
  return &m_ShapesWithTypeTool;
};


map<Standard_Integer, Standard_Integer>* 
GridBndData::
ModifyShapesType() 
{
  return &m_ShapesWithTypeTool;
};

const map<Standard_Integer, Standard_Integer>* 
GridBndData::
GetFacesType() const 
{
  return &m_FacesWithTypeTool;
};


map<Standard_Integer, Standard_Integer>* 
GridBndData::
ModifyFacesType() 
{
  return &m_FacesWithTypeTool;
};



const map<Standard_Integer, Standard_Integer>* 
GridBndData::
GetShapesMask() const 
{
  return &m_ShapeMaskWithIndexTool;
}; 

map<Standard_Integer, Standard_Integer>* 
GridBndData::
ModifyShapesMask() 
{
  return &m_ShapeMaskWithIndexTool;
}; 


const map<Standard_Integer, Standard_Integer>* 
GridBndData::
GetFacesMask() const 
{
  return &m_FaceMaskWithIndexTool;
}; 


map<Standard_Integer, Standard_Integer>* 
GridBndData::
ModifyFacesMask() 
{
  return &m_FaceMaskWithIndexTool;
}; 


Standard_Integer 
GridBndData::
GetBackGroundMaterialType() const
{

  return m_BackGroundMaterialType;
};

Standard_Integer 
GridBndData::
GetBackGroundMaterialDataIndex() const
{
  return m_BackGroundMaterialDataIndex;
}; 

void 
GridBndData::
SetBackGroundMaterialType(const Standard_Integer aType)
{ 
  m_BackGroundMaterialType = aType;
};




const map<Standard_Integer, Standard_Integer>* 
GridBndData::
GetRelationBetweenFaceAndShape() const 
{
  return &m_FacesWithShapeTool;
};


map<Standard_Integer, Standard_Integer>* 
GridBndData::
ModifyRelationBetweenFaceAndShape() 
{
  return &m_FacesWithShapeTool;
};


const map<Standard_Integer, vector<Standard_Integer> >* 
GridBndData::
GetRelationBetweenEdgeAndFace() const 
{
  return &m_EdgesWithFaceTool;
};


map<Standard_Integer, vector<Standard_Integer> >* 
GridBndData::
ModifyRelationBetweenEdgeAndFace() 
{
  return &m_EdgesWithFaceTool;
};


const map<Standard_Integer, vector<Standard_Integer> >* 
GridBndData::
GetRelationBetweenVertexAndEdge() const 
{
  return &m_VerticesWithEdgeTool;
};


map<Standard_Integer, vector<Standard_Integer> >* 
GridBndData::
ModifyRelationBetweenVertexAndEdge() 
{
  return &m_VerticesWithEdgeTool;
};
