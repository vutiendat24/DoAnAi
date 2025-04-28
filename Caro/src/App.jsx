import { useState, useEffect } from 'react';
import './App.css';

const BOARD_SIZE = 15;
const TIME_LIMIT = 20; // 20 giây mỗi lượt

function App() {
  const emptyBoard = Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(0));

  const [board, setBoard] = useState(emptyBoard);
  const [currentPlayer, setCurrentPlayer] = useState(1); // 1: Người (X), -1: Máy (O)
  const [winningCells, setWinningCells] = useState([]);
  const [timer, setTimer] = useState(TIME_LIMIT);

  useEffect(() => {
    if (winningCells.length > 0) return;

    // Máy tự đi khi tới lượt máy (O)
    if (currentPlayer === -1) {
      const delay = setTimeout(() => {
        makeAIMove();
      }, 500); // delay 0.5s cho tự nhiên

      return () => clearTimeout(delay);
    }
  }, [currentPlayer, winningCells]);

  useEffect(() => {
    if (winningCells.length > 0) return; // Nếu thắng thì dừng đếm thời gian

    const interval = setInterval(() => {
      setTimer((prev) => {
        if (prev === 1) {
          handleTimeout();
          return TIME_LIMIT;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [currentPlayer, winningCells]);

  const handleTimeout = () => {
    alert(`Hết giờ! Người chơi ${currentPlayer === 1 ? 'X' : 'O'} bị mất lượt.`);
    setCurrentPlayer((prev) => prev * -1); // Chuyển lượt
    setTimer(TIME_LIMIT);
  };

  const handleClick = (row, col) => {
    if (board[row][col] !== 0 || winningCells.length > 0 || currentPlayer !== 1) return; // Chỉ người chơi mới được click

    makeMove(row, col, 1);
  };

  const makeMove = (row, col, player) => {
    const newBoard = board.map((r) => [...r]);
    newBoard[row][col] = player;
    setBoard(newBoard);

    const winPositions = isWin(newBoard, row, col, player);
    if (winPositions) {
      setWinningCells(winPositions);
      setTimeout(() => {
        alert(`Người chơi ${player === 1 ? 'X' : 'O'} thắng!`);
      }, 100);
      return;
    }

    setCurrentPlayer(player * -1); // Chuyển lượt sau mỗi nước đi
    setTimer(TIME_LIMIT); // Reset lại thời gian cho lượt tiếp theo
  };

  const makeAIMove = async () => {
    if (currentPlayer !== -1) return; // Nếu không phải lượt của máy, không thực hiện
  
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ board }),
      });
  
      if (!response.ok) {
        console.error(`Error: ${response.status} - ${response.statusText}`);
        throw new Error('Không thể kết nối với máy chủ!');
      }
  
      const data = await response.json();
      console.log('AI move response:', data);
  
      if (data.error) {
        throw new Error(data.error);
      }
  
      const { row, col } = data;
  
      // Kiểm tra nếu vị trí đã được chọn, yêu cầu máy chọn lại vị trí khác
      if (row < 0 || col < 0 || board[row][col] !== 0) {
        console.log('Vị trí đã được chọn, máy sẽ chọn lại!');
        makeAIMove(); // Gọi lại hàm để máy chọn nước đi khác
        return;
      }
  
      // Thực hiện nước đi của máy
      makeMove(row, col, -1);
    } catch (error) {
      console.error('Lỗi gọi AI server:', error);
      alert('Lỗi kết nối với máy chủ AI, vui lòng thử lại!');
    }
  };
  const handleRestart = () => {
    setBoard(emptyBoard);
    setCurrentPlayer(1);
    setWinningCells([]);
    setTimer(TIME_LIMIT);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 to-purple-100 flex flex-col items-center justify-center p-4">
      <h1 className="text-4xl font-bold text-gray-800 mb-6">Caro Game (Người vs Máy)</h1>

      <div className="bg-white rounded-lg shadow-lg p-6 mb-6 w-full max-w-md">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-gray-700">
            Lượt: {currentPlayer === 1 ? 'Người (X)' : 'Máy (O)'}
          </h2>
          <h2 className={`text-xl font-semibold ${timer <= 5 ? 'text-red-500 animate-pulse' : 'text-gray-700'}`}>
            Thời gian: {timer}s
          </h2>
        </div>
      </div>

      <div className="board bg-white rounded-lg shadow-lg p-2">
        {board.map((row, rowIndex) => (
          <div key={rowIndex} className="flex">
            {row.map((cell, colIndex) => (
              <div
                key={colIndex}
                className={`w-10 h-10 flex items-center justify-center border border-gray-300 text-2xl font-bold cursor-pointer transition-colors
                  ${winningCells.some(([r, c]) => r === rowIndex && c === colIndex) ? 'bg-yellow-300 text-gray-800' : 'bg-gray-50 hover:bg-gray-100'}
                  ${cell === 1 ? 'text-blue-600' : cell === -1 ? 'text-red-600' : ''}`}
                onClick={() => handleClick(rowIndex, colIndex)}
              >
                {cell === 1 ? 'X' : cell === -1 ? 'O' : ''}
              </div>
            ))}
          </div>
        ))}
      </div>

      <button
        className="mt-6 px-6 py-2 bg-blue-600 text-blue-600 rounded-lg shadow-md hover:bg-blue-700 transition-colors"
        onClick={handleRestart}
      >
        Chơi lại
      </button>
    </div>
  );
}

function isWin(board, row, col, player) {
  const directions = [
    { dr: 0, dc: 1 },
    { dr: 1, dc: 0 },
    { dr: 1, dc: 1 },
    { dr: 1, dc: -1 },
  ];

  const BOARD_SIZE = board.length;

  for (let { dr, dc } of directions) {
    let positions = [[row, col]];

    let r = row + dr;
    let c = col + dc;
    while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && board[r][c] === player) {
      positions.push([r, c]);
      r += dr;
      c += dc;
    }

    r = row - dr;
    c = col - dc;
    while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && board[r][c] === player) {
      positions.push([r, c]);
      r -= dr;
      c -= dc;
    }

    if (positions.length >= 5) {
      return positions;
    }
  }

  return null;
}

export default App;
