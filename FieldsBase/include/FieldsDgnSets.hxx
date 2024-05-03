
#ifndef _FieldsDgnSets_HeaderFile
#define _FieldsDgnSets_HeaderFile

#include <FieldsDefineCntr.hxx>
#include <DynObj.hxx>

#include <TxHierAttribSet.h>
#include <iostream>

class FieldsDgnBase;


class FieldsDgnSets : public DynObj
{
public:
  FieldsDgnSets();
  FieldsDgnSets(const FieldsDefineCntr* commcntr);
  virtual ~FieldsDgnSets();


public:
  void SetAttrib(const TxHierAttribSet& tha);
  void Init(const FieldsDefineCntr* commcntr);

  void Append(FieldsDgnBase* _oneNewPort);

  const FieldsDefineCntr* GetFldsDefCntr() const;

  void Advance();

  void Dump(std::ostream& out);
  void DumpHead(std::ostream& out);


private:
  vector<FieldsDgnBase*> m_DgnGrp;

  const FieldsDefineCntr* m_FldsDefCntr;
};


#endif

