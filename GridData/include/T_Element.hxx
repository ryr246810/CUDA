#ifndef _T_Element_Headerfile
#define _T_Element_Headerfile

#include <Standard_TypeDefine.hxx>

class DataBase;

class T_Element
{
public:
  T_Element();
  T_Element(DataBase* _Data, Standard_Integer _Dir);
  ~T_Element();

public:
  void SetData(DataBase* _Data);
  void SetRelatedDir(Standard_Integer aDir);


  DataBase* GetData() const;
  Standard_Integer GetRelatedDir() const;

private:
  Standard_Integer m_RelatedDir;
  DataBase* m_Data;
};

#endif

