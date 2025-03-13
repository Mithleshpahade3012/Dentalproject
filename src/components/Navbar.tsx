import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Stethoscope, Home, Phone } from 'lucide-react';

const Navbar = () => {
  const location = useLocation();

  const isActive = (path: string) => {
    return location.pathname === path ? 'bg-gray-100' : '';
  };

  return (
    <nav className="fixed w-full bg-white/80 backdrop-blur-md text-gray-800 shadow-lg z-50">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center space-x-2">
            <Stethoscope className="h-8 w-8" />
            <span className="font-bold text-xl">Dental Health AI</span>
          </Link>
          <div className="flex space-x-4">
            <Link
              to="/"
              className={`flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium hover:bg-gray-100 transition-colors ${isActive(
                '/'
              )}`}
            >
              <Home className="h-4 w-4" />
              <span>Home</span>
            </Link>
            <Link
              to="/analysis"
              className={`flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium hover:bg-gray-100 transition-colors ${isActive(
                '/analysis'
              )}`}
            >
              <Stethoscope className="h-4 w-4" />
              <span>Analysis</span>
            </Link>
            <Link
              to="/contact"
              className={`flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium hover:bg-gray-100 transition-colors ${isActive(
                '/contact'
              )}`}
            >
              <Phone className="h-4 w-4" />
              <span>Contact</span>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;