#ifndef _Geom_TxtBuilders_HeaderFile
#define _Geom_TxtBuilders_HeaderFile


#include <Geom_TxtBuilderBase.hxx>

class Geom_TxtBuilders
{
public:
  Geom_TxtBuilders();
  ~Geom_TxtBuilders();


public:
  virtual void SetAttrib(const TxHierAttribSet& tha);
  void Init(OCAFDocumentCtrl* _ocafDocCtrl);
  void ClearBuilders();


protected:
  OCAFDocumentCtrl* m_DocCtrl;
  vector<Geom_TxtBuilderBase*> m_Builders;
};

#endif
