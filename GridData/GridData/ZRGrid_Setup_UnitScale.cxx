#include <ZRGrid.hxx>
#include <algorithm>


void 
ZRGrid::
ScaleAccordingUnitSystem(UnitsSystemDef* theUnitsSystem)
{
  for(Standard_Integer i = 0; i < 2; ++i){
    m_Org[i] = m_Org[i] * theUnitsSystem->GetRealUnitScaleOfLength();
  }

  map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer> >::iterator mapIter;

  for(mapIter = m_LVectors.begin(); mapIter!=m_LVectors.end(); mapIter++){
    vector<Standard_Real>& currLengthVec = mapIter->second;
    Standard_Size nb = currLengthVec.size();

    for(Standard_Size i = 0; i < nb; ++i){
      currLengthVec[i] = currLengthVec[i] * theUnitsSystem->GetRealUnitScaleOfLength();
    }
  }

  Setup();
}
