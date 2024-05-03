#ifndef _ComboFields_Dynamic_VertexElecSrc_HeaderFile
#define _ComboFields_Dynamic_VertexElecSrc_HeaderFile

#include <ComboFields_Dynamic_SrcBase.hxx>

#include <TxHierAttribSet.h>
#include <TFunc.hxx>


class ComboFields_Dynamic_VertexElecSrc : public ComboFields_Dynamic_SrcBase
{

public:
  ComboFields_Dynamic_VertexElecSrc();
  ComboFields_Dynamic_VertexElecSrc(const FieldsDefineCntr* theCntr, PhysDataDefineRule theRule);

  virtual ~ComboFields_Dynamic_VertexElecSrc();

  virtual void SetAttrib(const TxHierAttribSet& tha);


public:
  virtual void Setup();
  virtual void Advance();

private:
  void SetupRgn();


protected:
  Standard_Size m_VertexIndex[2];

  Standard_Real m_StartTime;
  Standard_Real m_EndTime;
  Standard_Real m_phiIndex;
  vector<GridVertexData*> m_Datas;

  TFunc* m_tfuncPtr;
};

#endif
