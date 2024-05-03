#include <GridFaceData.cuh>
#include <GridFace.hxx>
#include <T_Element.hxx>
#include <GridGeometry.hxx>
#include <cmath>

//#define GRIDFACEDATA_DBG
//#define AREA_DBG

#define AREA_REF_RATIO 0.4987654321
#define REF_A 0.5
#define LEFT_AREA_RATIO 1.0


/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/

bool 
GridFaceData::
IsSmallArea() const
{
  bool tmp = false;
  Standard_Real areaRatio = FaceAreaRatio();
  if(areaRatio<AREA_REF_RATIO) tmp = true;
  return tmp;
}

Standard_Real 
GridFaceData::
FaceAreaRatio() const
{
  Standard_Real tmpRatio = this->GetGeomDim()/this->GetBaseGridFace()->GetArea();

#ifdef GRIDFACEDATA_DBG
  cout<<"GridFaceData::FaceAreaRatio()--------The ratio is\t=\t"<<tmpRatio<<endl;
#endif

  return tmpRatio;
}

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
