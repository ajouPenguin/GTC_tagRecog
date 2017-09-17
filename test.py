# -*- coding: utf-8 -*-
# UTF-8 encoding when using korean


"""
@description
  하나의 점(좌표)를 나타내는 클래스
"""
class point:
  def __init__(self, x, y):
    self.x = x
    self.y = y


"""
@description
  두 사각형을 이루는 네 점을 파라미터로 받아 교차하는 영역의 넓이를 반환하는 함수
	각 점은 point class의 객체로 주어진다

@param p	첫 번째 사각형의 한 점, q와 대각선상에 존재한다.
@param q	첫 번째 사각형의 한 점, p와 대각선상에 존재한다.
@param r	두 번째 사각형의 한 점, s와 대각선상에 존재한다.
@param s	두 번째 사각형의 한 점, r과 대각선상에 존재한다.
@return   두 직사각형이 교차하는 영역의 넓이
"""
def get_duplicated_area(p, q, r, s):
  area = 0

  return area

"""
메인 함수에는 테스트케이스와 입출력에 대한 기본적인 뼈대 코드가 작성되어 있습니다. 
상단의 함수만 완성하여도 문제를 해결할 수 있으며, 
메인 함수를 제거한 후 스스로 코드를 모두 작성하여도 무방합니다.
단, 스스로 작성한 코드로 인해 발생한 에러 등은 모두 참가자에게 책임이 있습니다.
"""
if __name__ == "__main__":
  #네 점의 좌표를 입력받는다 
  px, py =[ int(word) for word in raw_input().split() ] 
  qx, qy =[ int(word) for word in raw_input().split() ] 
  rx, ry =[ int(word) for word in raw_input().split() ] 
  sx, sy =[ int(word) for word in raw_input().split() ] 

  #각 점의 정보를 객체화한다
  p = point(px, py)
  q = point(qx, qy)
  r = point(rx, ry)
  s = point(sx, sy)

  #주어진 함수를 통해 교차하는 영역의 넓이를 계산한다 
  answer = get_duplicated_area(p, q, r, s)

  #정답을 형식에 맞게 출력한다
  print(answer)