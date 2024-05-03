#include <Grid_Tool.hxx>

Grid_Tool::
Grid_Tool()
{

}


Grid_Tool::
~Grid_Tool()
{
  
}


void 
Grid_Tool::
SetGridCtrl(ZRGrid_Ctrl* theGC)
{
  m_GridCtrl = theGC;
}


void 
Grid_Tool::
SetAttrib(const TxHierAttribSet& tas)
{
  if(tas.hasOption("dir")){
    m_Dir = tas.getOption("dir");
  }
}


void 
Grid_Tool::
Build()
{

}


Standard_Integer 
Grid_Tool::
GetDir() const
{
  return m_Dir;
}


Standard_Real 
Grid_Tool::
GetFirstStep() const
{
  Standard_Real result = 0.0;
  if(!m_CoordinateVec.empty()){
    Standard_Size np = m_CoordinateVec.size();
    if(np>=2){
      result = m_CoordinateVec[1] - m_CoordinateVec[0];
    }
  }

  return result;
}


Standard_Real 
Grid_Tool::
GetLastStep() const
{
  Standard_Real result = 0.0;
  if(!m_CoordinateVec.empty()){
    Standard_Size np = m_CoordinateVec.size();
    if(np>=2){
      result = m_CoordinateVec[np-1] - m_CoordinateVec[np-2];
    }
  }

  return result;
}


const vector<Standard_Real>& 
Grid_Tool::
GetResult() const
{
  return m_CoordinateVec;
}


vector<Standard_Real>& 
Grid_Tool::
ModifyResult()
{
  return m_CoordinateVec;
}


Standard_Real 
Grid_Tool::
GetOrg() const
{
  Standard_Real result = 0.0;
  if(!m_CoordinateVec.empty()){
    result = m_CoordinateVec[0];
  }
  return result;
}
