#ifndef _AppendingVertexDataOfGridFace_Headerfile
#define _AppendingVertexDataOfGridFace_Headerfile

#include<GridFace.hxx>
#include<VertexData.hxx>

class AppendingVertexDataOfGridFace: public VertexData
{
public:
  AppendingVertexDataOfGridFace();

  AppendingVertexDataOfGridFace(Standard_Integer _shapeindex,
				Standard_Integer _edgeindex,
				const set<Standard_Integer>& _faceindices,
				GridFace* _baseface, 
				Standard_Size _frac1,
				Standard_Size _frac2,
				Standard_Integer _mark,
				Standard_Integer _materialtype);

  ~AppendingVertexDataOfGridFace();


  /******************* Set ************************/
public:
  void SetBaseGridFace(GridFace* f){ m_BaseFace = f; };
  void SetLocation( Standard_Size _Frac1, Standard_Size _Frac2 ){ m_Frac1 = _Frac2; m_Frac2 = _Frac2; };
  void SetFrac1(Standard_Size _frac){ m_Frac1 = _frac; };
  void SetFrac2(Standard_Size _frac){ m_Frac2 = _frac; };


  /******************* Get ************************/
public:
  GridFace*         GetBaseFace() const {return m_BaseFace;};
  Standard_Size     GetFrac1()const {return m_Frac1;};
  Standard_Size     GetFrac2()const {return m_Frac2;};
  Standard_Size     GetBaseFaceIndex() const{ return m_BaseFace->GetIndex(); };
  /************************************************/


public:
  bool IsSameLocation(const AppendingVertexDataOfGridFace* one);


public:
  virtual TxVector2D<Standard_Real> GetLocation() const;


private:
  Standard_Size     m_Frac1;
  Standard_Size     m_Frac2;

  GridFace*  m_BaseFace;
};

#endif
