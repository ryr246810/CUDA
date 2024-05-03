#ifndef _AppendingVertexDataOfGridEdge_Headerfile
#define _AppendingVertexDataOfGridEdge_Headerfile

#include<GridEdge.hxx>
#include<VertexData.hxx>

class AppendingVertexDataOfGridEdge:public VertexData
{
public:
  AppendingVertexDataOfGridEdge();

  AppendingVertexDataOfGridEdge(Standard_Integer _ShapeIndex, 
				Standard_Integer _FaceIndex,
				GridEdge* _baseedge,
				Standard_Size _frac,
				Standard_Integer _mark,
				Standard_Integer _materialtype,
				Standard_Integer  _transitiontype);

  ~AppendingVertexDataOfGridEdge();
  

  /******************* Set ************************/
public:
  void SetBaseGridEdge(GridEdge* e){ m_BaseEdge = e; };
  void SetLocation( Standard_Size _Frac ){ m_Frac = _Frac;  };
  void SetFrac(Standard_Size _frac){ m_Frac = _frac; };
  void SetTransitionType(Standard_Integer _transtype){ m_TransitionType = _transtype; };
  /************************************************/


  /******************* Get ************************/
public:
  GridEdge*         GetBaseEdge(){return m_BaseEdge;};
  Standard_Size     GetFrac(){return m_Frac;};
  Standard_Integer  GetTransitionType(){return m_TransitionType; };
  /************************************************/

  bool HasSameLocation(AppendingVertexDataOfGridEdge* oneData);

  /******************* Tool ***********************/
public:
  virtual TxVector2D<Standard_Real> GetLocation() const;

  /************************************************/

private:
  Standard_Size     m_Frac;
  Standard_Integer  m_TransitionType;
  GridEdge*         m_BaseEdge;
};

#endif
