#include <OCAF_ColorMap.hxx>
#include <Quantity_NameOfColor.hxx>


map<Standard_Integer, Quantity_NameOfColor, less<Standard_Integer> > OCAF_ColorMap::m_IndexWithColorTool;
map<Standard_Integer, string, less<Standard_Integer> > OCAF_ColorMap::m_IndexWithNameTool;
map<string, Standard_Integer, less<string> > OCAF_ColorMap::m_NameWithIndexTool;
map<Standard_Integer, Standard_Integer, less<Standard_Integer> > OCAF_ColorMap::m_IndexWithMaterialTypeTool;



Standard_Integer OCAF_ColorMap::getMaterialIndex(const string& theName,
						 Standard_Boolean& IsOk)
{
  Standard_Integer result=0;
  IsOk = Standard_False;

  map<string, Standard_Integer>::iterator iter = m_NameWithIndexTool.find(theName);
  if(iter!=m_NameWithIndexTool.end()){
    result = iter->second;
    IsOk = Standard_True;
  }
  return result;
}


string OCAF_ColorMap::getName(const Standard_Integer& theIndex,
			      Standard_Boolean& IsOk)
{
  string result=" ";
  IsOk = Standard_False;
  map<Standard_Integer, string>::iterator iter = m_IndexWithNameTool.find(theIndex);
  if(iter!=m_IndexWithNameTool.end()){
    result = iter->second;
    IsOk = Standard_True;
  }
  return result;
}


Quantity_NameOfColor OCAF_ColorMap::getColor(const Standard_Integer& theIndex,
					     Standard_Boolean& IsOk)
{
  //Quantity_NameOfColor result = Quantity_NOC_GOLDENROD;
  //Quantity_NameOfColor result = Quantity_NOC_GRAY50;

  Quantity_NameOfColor result = Quantity_NOC_WHEAT;

  IsOk = Standard_False;

  map<Standard_Integer, Quantity_NameOfColor>::iterator iter = m_IndexWithColorTool.find(theIndex);
  if(iter!=m_IndexWithColorTool.end()){
    result = iter->second;
    IsOk = Standard_True;
  }

  return result;
}


Standard_Integer OCAF_ColorMap::GetMaterialType(const Standard_Integer& theIndex, 
						Standard_Boolean& IsOk)
{
  Standard_Integer result = 0;
  IsOk = Standard_False;

  map<Standard_Integer, Standard_Integer>::iterator iter = m_IndexWithMaterialTypeTool.find(theIndex);

  if(iter!=m_IndexWithMaterialTypeTool.end()){
    result = iter->second;
    IsOk = Standard_True;
  }

  return result;
}


void OCAF_ColorMap::InsertColorDefine(const Standard_Integer& theIndex,
				      const string& theName,
				      const Standard_Integer& theMaterialType,
				      const Quantity_NameOfColor& theColor,
				      Standard_Boolean& IsOk)
{
  IsOk = Standard_False;

  map<Standard_Integer, Quantity_NameOfColor>::iterator iter_Color = m_IndexWithColorTool.find(theIndex);
  map<Standard_Integer, Standard_Integer>::iterator iter_MaterialType = m_IndexWithMaterialTypeTool.find(theIndex);
  map<Standard_Integer, string>::iterator iter_Name = m_IndexWithNameTool.find(theIndex);
  map<string, Standard_Integer>::iterator iter_Index = m_NameWithIndexTool.find(theName);


  if( iter_Color==m_IndexWithColorTool.end()  && 
      iter_Name==m_IndexWithNameTool.end() &&
      iter_MaterialType==m_IndexWithMaterialTypeTool.end() &&
      iter_Index==m_NameWithIndexTool.end()){

    m_NameWithIndexTool.insert( pair<string, Standard_Integer>(theName, theIndex) );
    m_IndexWithNameTool.insert( pair<Standard_Integer, string>(theIndex, theName) );
    m_IndexWithMaterialTypeTool.insert( pair<Standard_Integer, Standard_Integer>(theIndex, theMaterialType));
    m_IndexWithColorTool.insert( pair<Standard_Integer, Quantity_NameOfColor>(theIndex, theColor) );


    IsOk = Standard_True;
  }
}


void OCAF_ColorMap::ClearColorDefine()
{
  m_IndexWithColorTool.clear();
  m_IndexWithNameTool.clear();
  m_NameWithIndexTool.clear();
  m_IndexWithMaterialTypeTool.clear();
}


Standard_Size OCAF_ColorMap::getSize()
{
  return m_IndexWithColorTool.size();
}
