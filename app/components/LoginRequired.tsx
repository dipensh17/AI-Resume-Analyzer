import React from 'react';
import { usePuterStore } from '~/lib/puter';

const LoginRequired: React.FC = () => {
    const { auth, loginRequired, clearLoginRequired, isLoading } = usePuterStore();

    if (!loginRequired) return null;

    const handleLogin = async () => {
        await auth.signIn();
        clearLoginRequired();
    };

    const handleDismiss = () => {
        clearLoginRequired();
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
            <div className="bg-white rounded-lg p-8 w-[90%] max-w-md text-center shadow-lg">
                <h2 className="text-2xl font-semibold">Login required</h2>
                <p className="mt-4 text-gray-600">Your session has expired or you need to sign in to continue.</p>
                <div className="mt-6 flex gap-3 justify-center">
                    <button className="primary-button" onClick={handleLogin} disabled={isLoading}>
                        {isLoading ? 'Signing in...' : 'Sign In'}
                    </button>
                    <button className="secondary-button" onClick={handleDismiss}>
                        Dismiss
                    </button>
                </div>
            </div>
        </div>
    );
};

export default LoginRequired;
