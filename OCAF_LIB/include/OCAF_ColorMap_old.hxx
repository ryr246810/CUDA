#ifndef _OCAF_ColorMap_HeaderFile
#define _OCAF_ColorMal_HeaderFile

#include <Quantity_NameOfColor.hxx>
#include <Standard_Integer.hxx>
#include <string>
#include <map>

using namespace std;

class OCAF_ColorMap
{

public:
  OCAF_ColorMap(){};
  ~OCAF_ColorMap(){};


public:
  static void InsertColorDefine(const Standard_Integer& theIndex,
				const string& theName,
				const Standard_Integer& theMaterialType,
				const Quantity_NameOfColor& theColor,
				Standard_Boolean& IsOk);

  static void ClearColorDefine();


public:
  static Standard_Size getSize();

  static Quantity_NameOfColor getColor(const Standard_Integer& theIndex,
				       Standard_Boolean& IsOk);

  static string getName(const Standard_Integer& theIndex,
			Standard_Boolean& IsOk);

  static Standard_Integer GetMaterialType(const Standard_Integer& theIndex, 
					  Standard_Boolean& IsOk);

private:
  static map<Standard_Integer, Quantity_NameOfColor, less<Standard_Integer> > m_IndexWithColorTool;
  static map<Standard_Integer, string, less<Standard_Integer> > m_IndexWithNameTool;
  static map<Standard_Integer, Standard_Integer, less<Standard_Integer> > m_IndexWithMaterialTypeTool;
};

#endif

