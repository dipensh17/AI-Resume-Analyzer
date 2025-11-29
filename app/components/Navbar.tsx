import { Link } from "react-router";
import { usePuterStore } from "~/lib/puter";

const Navbar = () => {
    const { auth } = usePuterStore();
    const username = auth.user?.username || "User";
    const handleLogout = async () => {
        await auth.signOut();
        window.location.href = "/auth";
    };
    return (
        <nav className="navbar">
            <Link to="/">
                <p className="text-2xl font-bold text-gradient">RESUMIND</p>
            </Link>
            <div className="flex items-center gap-6">
                <Link to="/upload" className="primary-button w-fit">
                    Upload Resume
                </Link>
                <div className="flex items-center gap-3 user-info">
                    {/* User Icon */}
                    <span className="inline-block w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 text-gray-600">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 7.5a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z" />
                            <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 19.5a7.5 7.5 0 1115 0v.75A2.25 2.25 0 0117.75 22.5H6.25A2.25 2.25 0 014 20.25v-.75z" />
                        </svg>
                    </span>
                    <span className="font-medium text-dark-200">{username}</span>
                    <button className="primary-button px-3 py-1 text-sm" onClick={handleLogout}>Logout</button>
                </div>
            </div>
        </nav>
    )
}
export default Navbar
