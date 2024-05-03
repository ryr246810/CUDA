#ifndef _ComboFieldsDefineRules_HeaderFile
#define _ComboFieldsDefineRules_HeaderFile

#include <Standard_TypeDefine.hxx>
#include <FieldsDefineRules.hxx>
#include <set>
#include <vector>
#include <map>

class ComboFieldsDefineRules : public FieldsDefineRules
{
public:
  ComboFieldsDefineRules();
  virtual ~ComboFieldsDefineRules();

public:
  virtual void Setup_Fields_PhysDatasNum_AccordingMaterialDefine();
};

#endif
