
#include <ZRDefine.hxx>


void 
ZRDefine::
ScaleAccordingUnitSystem(UnitsSystemDef* theUnitsSystem)
{
  for(Standard_Integer i = 0; i < 3; ++i){
    m_XYZOrg[i] =  m_XYZOrg[i] * theUnitsSystem->GetRealUnitScaleOfLength();
  }
}
