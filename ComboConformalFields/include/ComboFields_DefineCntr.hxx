#ifndef _ComboFields_DefineCntr_HeaderFile
#define _ComboFields_DefineCntr_HeaderFile

#include <GridGeometry.hxx>
#include <GridGeometry_Cyl3D.hxx>
#include <FieldsDefineCntr.hxx>
#include <ComboFieldsDefineRules.hxx>

class ComboFields_DefineCntr : public FieldsDefineCntr
{
public:
  ComboFields_DefineCntr(const GridGeometry* gridGeom, 
			 const ComboFieldsDefineRules* defrules);

  virtual ~ComboFields_DefineCntr();

public:
  virtual void LocateMemeory_For_FieldsPhysDatas();

  void LocateMemeory_For_3DFieldsPhysDatas();


private:
  void LocateMemeory_For_VertexPhysDatas();
  void LocateMemeory_For_EdgePhysDatas();
  void LocateMemeory_For_FacePhysDatas();

  void LocateMemeory_For_3DVertexPhysDatas();
  void LocateMemeory_For_3DEdgePhysDatas();
  void LocateMemeory_For_3DFacePhysDatas();

  // prevent using the default constructor
private:
  ComboFields_DefineCntr();



};

#endif
